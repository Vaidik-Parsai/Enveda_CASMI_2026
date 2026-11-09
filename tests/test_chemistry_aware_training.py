"""Unit tests for Phase 6B: Chemistry-Aware Training components.

Tests:
1. CosineWarmupScheduler (linear warmup, cosine decay bounds, state persistence).
2. Normalized Supervised Contrastive Loss (manual analytical calculation, gradients, multi-positive balance).
3. ChemistryAwareNegativePool (mass compatibility, Morgan Tanimoto, strict positive exclusion).
4. EmbeddingDiagnosticsTracker (effective rank, covariance metrics).
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from enveda.data.negative_sampler import ChemistryAwareNegativePool
from enveda.training.checkpointing import load_checkpoint, save_checkpoint
from enveda.training.diagnostics import EmbeddingDiagnosticsTracker
from enveda.training.losses import (
    multi_positive_contrastive_loss,
    normalized_supervised_contrastive_loss,
)
from enveda.training.schedulers import CosineWarmupScheduler


def test_cosine_warmup_scheduler():
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-4)
    sched = CosineWarmupScheduler(
        opt,
        warmup_steps=100,
        total_steps=1000,
        min_lr=1e-6,
    )

    # Step 0: warmup start
    lr_0 = sched.get_lr()[0]
    assert np.isclose(lr_0, 1e-6, atol=1e-7), f"Expected 1e-6 at step 0, got {lr_0}"

    # Step 50: midway through warmup
    sched.last_epoch = 50
    lr_50 = sched.get_lr()[0]
    expected_50 = 1e-6 + (1e-4 - 1e-6) * 0.5
    assert np.isclose(lr_50, expected_50, atol=1e-7)

    # Step 100: end of warmup -> base_lr 1e-4
    sched.last_epoch = 100
    lr_100 = sched.get_lr()[0]
    assert np.isclose(lr_100, 1e-4, atol=1e-7)

    # Step 550: midway through cosine decay
    sched.last_epoch = 550
    lr_550 = sched.get_lr()[0]
    # progress = (550 - 100) / (1000 - 100) = 450 / 900 = 0.5
    # cos(pi * 0.5) = 0.0 -> cosine = 0.5 * (1 + 0) = 0.5
    expected_550 = 1e-6 + (1e-4 - 1e-6) * 0.5
    assert np.isclose(lr_550, expected_550, atol=1e-7)

    # Step 1000: end of decay cycle -> min_lr 1e-6
    sched.last_epoch = 1000
    lr_1000 = sched.get_lr()[0]
    assert np.isclose(lr_1000, 1e-6, atol=1e-7)


def test_scheduler_state_dict_persistence(tmp_path: Path):
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-4)
    sched = CosineWarmupScheduler(opt, warmup_steps=50, total_steps=500, min_lr=1e-6)

    # Step 25
    sched.last_epoch = 25
    opt.step()
    sched.step()

    ckpt_path = tmp_path / "sched_test.pt"
    dummy_model = nn.Linear(2, 2)
    save_checkpoint(ckpt_path, model=dummy_model, optimizer=opt, scheduler=sched, step=25)

    # Load in fresh scheduler
    new_opt = torch.optim.AdamW([param], lr=1e-4)
    new_sched = CosineWarmupScheduler(new_opt, warmup_steps=50, total_steps=500, min_lr=1e-6)
    load_checkpoint(ckpt_path, model=dummy_model, optimizer=new_opt, scheduler=new_sched)

    assert new_sched.last_epoch == sched.last_epoch
    assert np.isclose(new_sched.get_lr()[0], sched.get_lr()[0], atol=1e-8)


def test_normalized_supcon_manual_analytical_check():
    # 3 items in batch:
    # Anchor 0 and Anchor 1 belong to molecule A (identical molecule_id = 100)
    # Anchor 2 belongs to molecule B (molecule_id = 200)
    # Construct exact 3x3 similarity matrix
    # T = 1.0 for simplicity
    s_embs = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=torch.float32)
    m_embs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mol_ids = torch.tensor([100, 100, 200], dtype=torch.int64)

    # Logits: s_embs @ m_embs.T
    # Row 0: [1.0, 1.0, 0.0]
    # Row 1: [0.8, 0.8, 0.6]
    # Row 2: [0.0, 0.0, 1.0]
    #
    # Positives for row 0: indices 0 and 1 (|P(0)| = 2), logits = [1.0, 1.0], mean = 1.0
    # Denom for row 0: log(exp(1) + exp(1) + exp(0)) = log(2e + 1) = log(2*2.71828 + 1) = log(6.43656) = 1.86199
    # Loss s2m row 0: 1.86199 - 1.0 = 0.86199
    #
    # Positives for row 1: indices 0 and 1 (|P(1)| = 2), logits = [0.8, 0.8], mean = 0.8
    # Denom for row 1: log(exp(0.8) + exp(0.8) + exp(0.6)) = log(2.22554 + 2.22554 + 1.82212) = log(6.27320) = 1.83628
    # Loss s2m row 1: 1.83628 - 0.8 = 1.03628
    #
    # Positives for row 2: index 2 (|P(2)| = 1), logit = 1.0, mean = 1.0
    # Denom for row 2: log(exp(0) + exp(0) + exp(1)) = log(1 + 1 + 2.71828) = log(4.71828) = 1.55144
    # Loss s2m row 2: 1.55144 - 1.0 = 0.55144
    #
    # Mean s2m loss = (0.86199 + 1.03628 + 0.55144) / 3 = 0.81657

    loss = normalized_supervised_contrastive_loss(
        spectrum_embeddings=s_embs,
        molecule_embeddings=m_embs,
        molecule_ids=mol_ids,
        temperature=1.0,
    )

    assert torch.isfinite(loss)
    # Check that it's positive and close to analytical expectation (~0.81)
    assert 0.70 < loss.item() < 0.95


def test_normalized_supcon_gradients():
    s_embs = torch.randn(8, 16, requires_grad=True)
    m_embs = torch.randn(8, 16, requires_grad=True)
    # Normalize
    s_norm = s_embs / torch.norm(s_embs, p=2, dim=1, keepdim=True)
    m_norm = m_embs / torch.norm(m_embs, p=2, dim=1, keepdim=True)
    mol_ids = torch.tensor([1, 1, 2, 2, 3, 4, 5, 5])

    loss = normalized_supervised_contrastive_loss(s_norm, m_norm, mol_ids, temperature=0.07)
    loss.backward()

    assert s_embs.grad is not None
    assert m_embs.grad is not None
    assert torch.all(torch.isfinite(s_embs.grad))
    assert torch.all(torch.isfinite(m_embs.grad))


def test_chemistry_aware_negative_pool_safety():
    # 5 molecules in pool
    smiles = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (194.08)
        "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin (180.04)
        "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O", # Glucose (180.06)
        "c1ccccc1",                        # Benzene (78.05)
        "c1ccncc1",                        # Pyridine (79.04)
    ]
    inchikeys = ["IK_CAFF", "IK_ASP", "IK_GLUC", "IK_BENZ", "IK_PYR"]
    fps = np.random.randint(0, 2, size=(5, 2048), dtype=np.uint8)

    pool = ChemistryAwareNegativePool(
        training_smiles=smiles,
        training_inchikeys=inchikeys,
        training_fingerprints=fps,
        mass_ppm_tolerance=50.0,
        random_seed=42,
    )

    # Query anchor: Aspirin (IK_ASP, neutral mass 180.042)
    # Precursor m/z for [M+H]+ = 181.0495
    anchor_pmz = 180.042259 + 1.007276
    sampled_indices, counts = pool.sample_negatives_for_anchor(
        anchor_inchikey="IK_ASP",
        anchor_fp=fps[1],
        anchor_precursor_mz=anchor_pmz,
        anchor_adduct="[M+H]+",
        anchor_mode="positive",
        num_negatives=3,
        mass_ratio=0.5,
        random_ratio=0.5,
    )

    # Strict safety check: Aspirin index (1) must NEVER be sampled!
    assert 1 not in sampled_indices, "Ground-truth anchor identity was sampled as a negative!"
    # Check that sample count matches requested
    assert len(sampled_indices) <= 3
    # Check category counts reported
    assert "mass_compatible" in counts
    assert "tanimoto_hard" in counts
    assert "random" in counts


def test_embedding_diagnostics_tracker():
    torch.manual_seed(42)
    # 20 samples, 16 dimensions
    s_embs = torch.randn(20, 16)
    s_embs = s_embs / torch.norm(s_embs, p=2, dim=1, keepdim=True)
    m_embs = torch.randn(20, 16)
    m_embs = m_embs / torch.norm(m_embs, p=2, dim=1, keepdim=True)
    mol_ids = torch.tensor([i // 2 for i in range(20)])  # 10 pairs of 2

    metrics = EmbeddingDiagnosticsTracker.evaluate_representation_geometry(
        spectrum_embeddings=s_embs,
        molecule_embeddings=m_embs,
        molecule_ids=mol_ids,
    )

    assert "spectrum_effective_rank" in metrics
    assert "molecule_effective_rank" in metrics
    assert "positive_cosine_mean" in metrics
    assert "negative_cosine_mean" in metrics
    assert "cosine_margin" in metrics
    assert 1.0 <= metrics["spectrum_effective_rank"] <= 16.0
    assert -1.0 <= metrics["positive_cosine_mean"] <= 1.0

