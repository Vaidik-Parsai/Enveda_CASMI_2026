"""Pre-training integration verification and smoke tests for Phase 6B."""

import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enveda.config import load_config
from enveda.data import (
    ChemistryAwareNegativePool,
    MetadataPreprocessor,
    StreamingParquetDataset,
    create_or_load_group_split,
)
from enveda.models import SpectrumMoleculeModel
from enveda.training import (
    CosineWarmupScheduler,
    Trainer,
    load_checkpoint,
    save_checkpoint,
)
from enveda.training.losses import (
    multi_positive_contrastive_loss,
    normalized_supervised_contrastive_loss,
)
from enveda.utils import setup_logger


def run_integration_verification():
    logger = setup_logger("integration_verification")
    logger.info("=" * 80)
    logger.info("PHASE 6B: PRE-TRAINING INTEGRATION VERIFICATION & SMOKE TESTS")
    logger.info("=" * 80)

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    train_rows, val_rows = create_or_load_group_split(train_parquet, split_path)
    logger.info(f"Loaded split: {len(train_rows):,} train rows | {len(val_rows):,} val rows")

    preprocessor = MetadataPreprocessor.load(config.data.meta_preprocessor_path)

    # -------------------------------------------------------------------------
    # 1. NEGATIVE SAMPLER INTEGRATION TRACE
    # -------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("1. TRACING COMPLETE NEGATIVE SAMPLER INTEGRATION PATH")
    logger.info("-" * 80)

    # Build negative pool strictly from a subset of training rows for verification
    logger.info("Building ChemistryAwareNegativePool on training molecules...")
    table = pq.read_table(
        str(train_parquet),
        columns=["normalized_smiles", "inchikey14", "precursor_mz", "adduct", "ionization_mode"],
    )
    df_sample = table.to_pandas().iloc[train_rows[:15000]].reset_index(drop=True)
    unique_train_df = df_sample.drop_duplicates(subset=["inchikey14"]).reset_index(drop=True)

    from enveda.data.preprocessing import generate_morgan_fingerprint
    pool_smiles = unique_train_df["normalized_smiles"].tolist()
    pool_inchikeys = unique_train_df["inchikey14"].tolist()
    pool_fps = np.array([generate_morgan_fingerprint(s)[0] for s in pool_smiles], dtype=np.uint8)

    neg_pool = ChemistryAwareNegativePool(
        training_smiles=pool_smiles,
        training_inchikeys=pool_inchikeys,
        training_fingerprints=pool_fps,
        tanimoto_threshold=0.3,
        mass_ppm_tolerance=50.0,
        random_seed=42,
    )
    logger.info(f"Initialized pool with {neg_pool.num_molecules:,} training molecules.")

    # Initialize model from Step 5,000 checkpoint
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)
    load_checkpoint("checkpoints/spectrum_molecule_step_5000.pt", model=model, device=device)
    model.eval()

    # Create batch generator
    train_row_set = set(train_rows)
    dataset = StreamingParquetDataset(
        parquet_path=str(train_parquet),
        meta_preprocessor=preprocessor,
        batch_size=128,
        row_indices_filter=train_row_set,
        spectrum_threshold=config.data.spectrum_threshold,
        spectrum_top_n=config.data.spectrum_top_n,
        bin_size=config.data.spectrum_bin_size,
        max_mz=config.data.spectrum_max_mz,
        num_bins=config.model.spectrum_dim,
        fp_radius=config.data.fingerprint_radius,
        fp_n_bits=config.model.fingerprint_dim,
        device=device,
    )

    batch_iter = iter(dataset)
    test_batch = next(batch_iter)

    # Trace step with num_negatives = 2
    inchikeys = test_batch["inchikeys"]
    pmzs = test_batch["precursor_mzs"]
    adducts = test_batch["adducts"]
    modes = test_batch["ionization_modes"]
    fps_np = test_batch["fingerprint"].cpu().numpy()

    # Track GPU memory before
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)

    sampled_neg_indices = set()
    category_counts_total = {"mass_compatible": 0, "tanimoto_hard": 0, "random": 0}
    false_positives_detected = 0

    for idx_a in range(len(inchikeys)):
        neg_idxs, counts = neg_pool.sample_negatives_for_anchor(
            anchor_inchikey=inchikeys[idx_a],
            anchor_fp=fps_np[idx_a],
            anchor_precursor_mz=float(pmzs[idx_a]),
            anchor_adduct=adducts[idx_a],
            anchor_mode=modes[idx_a],
            num_negatives=2,
            anchor_idx=idx_a,
            mass_ratio=0.50,
            tanimoto_ratio=0.25,
            random_ratio=0.25,
        )
        sampled_neg_indices.update(neg_idxs)
        for cat, cnt in counts.items():
            category_counts_total[cat] += cnt

        # Verify safety: none of the sampled negatives match anchor identity
        for n_i in neg_idxs:
            n_ikey = neg_pool.inchikeys[n_i]
            if n_ikey == inchikeys[idx_a]:
                false_positives_detected += 1

    unique_neg_ids = len(sampled_neg_indices)
    assert false_positives_detected == 0, f"Detected {false_positives_detected} false negatives!"

    # Encode external negatives on GPU
    neg_fps_tensor = torch.from_numpy(neg_pool.fingerprints[list(sampled_neg_indices)]).float().to(device, non_blocking=True)
    with torch.no_grad():
        ext_neg_embs = model.encode_molecule(neg_fps_tensor)
        spec_emb, mol_emb = model(test_batch["spectrum"], test_batch["metadata"], test_batch["fingerprint"])

        loss_with_negs = normalized_supervised_contrastive_loss(
            spectrum_embeddings=spec_emb,
            molecule_embeddings=mol_emb,
            molecule_ids=test_batch["molecule_ids"],
            negative_molecule_embeddings=ext_neg_embs,
        )
        loss_without_negs = normalized_supervised_contrastive_loss(
            spectrum_embeddings=spec_emb,
            molecule_embeddings=mol_emb,
            molecule_ids=test_batch["molecule_ids"],
            negative_molecule_embeddings=None,
        )

    if device.type == "cuda":
        mem_after = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        mem_impact = mem_after - mem_before
    else:
        mem_impact = 0.0

    total_sampled = sum(category_counts_total.values())
    logger.info("Negative Sampler Integration Report (num_negatives = 2 per anchor):")
    logger.info(f"  Total negatives sampled across batch: {total_sampled} ({total_sampled/len(inchikeys):.1f} per anchor)")
    logger.info(f"  Category Proportions:")
    logger.info(f"    - Mass-Compatible (<= 50 ppm): {category_counts_total['mass_compatible']} ({category_counts_total['mass_compatible']/total_sampled*100:.1f}%)")
    logger.info(f"    - Morgan Tanimoto Structural:  {category_counts_total['tanimoto_hard']} ({category_counts_total['tanimoto_hard']/total_sampled*100:.1f}%)")
    logger.info(f"    - Random negatives:             {category_counts_total['random']} ({category_counts_total['random']/total_sampled*100:.1f}%)")
    logger.info(f"  Unique negative molecules encoded: {unique_neg_ids}")
    logger.info(f"  False negatives detected: {false_positives_detected} (Verified: 0)")
    logger.info(f"  GPU Memory Impact of external negatives: {mem_impact:.2f} MB")
    logger.info(f"  Contrastive Loss without external negatives: {loss_without_negs.item():.4f}")
    logger.info(f"  Contrastive Loss with external negatives:    {loss_with_negs.item():.4f}")
    assert loss_with_negs.item() > loss_without_negs.item(), "External negatives did not increase denominator penalty!"
    logger.info("  [VERIFIED] Sampled hard negatives actively participate in the loss denominator.")

    # Also test num_negatives = 4
    counts_4 = {"mass_compatible": 0, "tanimoto_hard": 0, "random": 0}
    for idx_a in range(len(inchikeys)):
        _, c4 = neg_pool.sample_negatives_for_anchor(
            anchor_inchikey=inchikeys[idx_a],
            anchor_fp=fps_np[idx_a],
            anchor_precursor_mz=float(pmzs[idx_a]),
            anchor_adduct=adducts[idx_a],
            anchor_mode=modes[idx_a],
            num_negatives=4,
            anchor_idx=idx_a,
        )
        for cat, cnt in c4.items():
            counts_4[cat] += cnt
    tot_4 = sum(counts_4.values())
    logger.info(f"Negative Sampler with num_negatives = 4 (Total: {tot_4}):")
    logger.info(f"    - Mass-Compatible: {counts_4['mass_compatible']} ({counts_4['mass_compatible']/tot_4*100:.1f}%)")
    logger.info(f"    - Morgan Tanimoto: {counts_4['tanimoto_hard']} ({counts_4['tanimoto_hard']/tot_4*100:.1f}%)")
    logger.info(f"    - Random:          {counts_4['random']} ({counts_4['random']/tot_4*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 2. SCHEDULER INTEGRATION CHECK
    # -------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("2. VERIFYING SCHEDULER WARMUP, DECAY, AND RESUME BEHAVIOUR")
    logger.info("-" * 80)

    dummy_param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([dummy_param], lr=1e-4)
    sched = CosineWarmupScheduler(opt, warmup_steps=500, total_steps=5000, min_lr=1e-6)

    # Detailed schedule verification at milestone steps
    milestone_targets = {
        0: 1.000e-6,
        1: 1.198e-6,
        2: 1.396e-6,
        10: 2.980e-6,
        250: 5.050e-5,
        500: 1.000e-4,
        2750: 5.050e-5,
        5000: 1.000e-6,
    }

    logger.info("Verifying exact scheduler learning rates across training steps:")
    for step_idx in range(5001):
        current_lr = opt.param_groups[0]["lr"]
        if step_idx in milestone_targets:
            target = milestone_targets[step_idx]
            assert np.isclose(current_lr, target, rtol=1e-3, atol=1e-7), (
                f"Step {step_idx}: expected LR {target:.7f}, got {current_lr:.7f}"
            )
            logger.info(f"  Step {step_idx:4d} | LR: {current_lr:.8f} (Expected: {target:.8f}) [OK]")
        sched.step()

    # Test state dict roundtrip
    ckpt_test_path = Path("checkpoints/smoke_test_sched.pt")
    ckpt_test_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scheduler_state_dict": sched.state_dict()}, str(ckpt_test_path))

    new_opt = torch.optim.AdamW([dummy_param], lr=1e-4)
    new_sched = CosineWarmupScheduler(new_opt, warmup_steps=500, total_steps=5000, min_lr=1e-6)
    loaded_sd = torch.load(str(ckpt_test_path), weights_only=False)["scheduler_state_dict"]
    new_sched.load_state_dict(loaded_sd)
    assert new_sched.last_epoch == sched.last_epoch
    assert np.isclose(new_sched.get_lr()[0], sched.get_lr()[0], atol=1e-8)
    ckpt_test_path.unlink()
    logger.info("  [VERIFIED] Scheduler state_dict correctly saves and restores last_epoch and learning rate.")

    # -------------------------------------------------------------------------
    # 3. TRAINING PIPELINE SMOKE TESTS (RUNS A, B, C, D)
    # -------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("3. RUNNING SYNTHETIC TRAINING SMOKE TESTS (RUNS A, B, C, D)")
    logger.info("-" * 80)

    smoke_dir = Path("checkpoints/smoke_test")
    smoke_dir.mkdir(parents=True, exist_ok=True)

    configs_to_test = [
        ("Run A", "multi_positive", None, 0, "Baseline MIL-NCE + In-Batch Random Negatives"),
        ("Run B", "normalized_supcon", None, 0, "Normalized SupCon + In-Batch Random Negatives"),
        ("Run C", "multi_positive", neg_pool, 2, "Baseline MIL-NCE + Chemistry-Aware External Negatives"),
        ("Run D", "normalized_supcon", neg_pool, 2, "Normalized SupCon + Chemistry-Aware External Negatives"),
    ]

    for run_name, loss_type, pool_obj, negs_per_anchor, desc in configs_to_test:
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING SMOKE TEST: {run_name} ({desc})")
        logger.info(f"{'='*70}")
        logger.info(f"  Configuration details:")
        logger.info(f"    - Loss objective:    {loss_type}")
        logger.info(f"    - Negative pool:     {'Enabled' if pool_obj is not None else 'Disabled'}")
        logger.info(f"    - Negatives/anchor:  {negs_per_anchor}")

        # Initialize fresh model from step 5k
        m = SpectrumMoleculeModel(
            spectrum_dim=config.model.spectrum_dim,
            metadata_dim=config.model.metadata_dim,
            fingerprint_dim=config.model.fingerprint_dim,
            embedding_dim=config.model.embedding_dim,
        ).to(device)
        load_checkpoint("checkpoints/spectrum_molecule_step_5000.pt", model=m, device=device)

        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        sched = CosineWarmupScheduler(opt, warmup_steps=500, total_steps=5000, min_lr=1e-6)

        tr_cfg = dict(config)
        tr_cfg["training"]["checkpoint_dir"] = str(smoke_dir)
        tr_cfg["training"]["checkpoint_interval"] = 3
        tr_cfg["training"]["max_steps"] = 3
        tr_cfg["training"]["scaler_init_scale"] = 8192.0

        trainer = Trainer(
            model=m,
            optimizer=opt,
            scheduler=sched,
            negative_pool=pool_obj,
            loss_fn_type=loss_type,
            negatives_per_anchor=negs_per_anchor,
            config=tr_cfg,
            device=device,
            logger=logger,
        )

        # Run 3 training steps and log exact optimizer learning rate progression
        losses = []
        lrs_progression = []
        for step_idx in range(3):
            lr_before_step = opt.param_groups[0]["lr"]
            loss_val = trainer.train_step(test_batch)
            trainer.global_step += 1
            lr_after_step = opt.param_groups[0]["lr"]
            losses.append(loss_val)
            lrs_progression.append((lr_before_step, lr_after_step))
            logger.info(
                f"    Step {step_idx + 1}/3: Loss = {loss_val:.4f} | "
                f"LR: {lr_before_step:.8f} -> {lr_after_step:.8f} (last_epoch={sched.last_epoch})"
            )

        assert sched.last_epoch == 3, f"Scheduler last_epoch {sched.last_epoch} != 3"
        assert losses[0] > 0 and losses[-1] > 0

        # Test checkpoint save and reload
        ckpt_p = smoke_dir / f"smoke_{run_name.lower().replace(' ', '_')}.pt"
        save_checkpoint(ckpt_p, model=m, optimizer=opt, scheduler=sched, step=3)

        # Reload into new model & scheduler
        new_m = SpectrumMoleculeModel(
            spectrum_dim=config.model.spectrum_dim,
            metadata_dim=config.model.metadata_dim,
            fingerprint_dim=config.model.fingerprint_dim,
            embedding_dim=config.model.embedding_dim,
        ).to(device)
        new_opt = torch.optim.AdamW(new_m.parameters(), lr=1e-4)
        new_sched = CosineWarmupScheduler(new_opt, warmup_steps=500, total_steps=5000, min_lr=1e-6)

        load_checkpoint(ckpt_p, model=new_m, optimizer=new_opt, scheduler=new_sched, device=device)
        assert new_sched.last_epoch == 3
        assert np.isclose(new_sched.get_lr()[0], sched.get_lr()[0], atol=1e-8)
        logger.info(f"  [PASS] {run_name} checkpoint save/reload verified (last_epoch={new_sched.last_epoch}, lr={new_sched.get_lr()[0]:.8f}).")

    # Clean up smoke test directory
    shutil.rmtree(smoke_dir, ignore_errors=True)
    logger.info("\nCleaned up temporary smoke test directory.")
    logger.info("=" * 80)
    logger.info("ALL INTEGRATION VERIFICATIONS & SMOKE TESTS PASSED CLEANLY.")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_integration_verification()
