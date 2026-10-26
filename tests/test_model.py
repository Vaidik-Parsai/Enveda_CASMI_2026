"""Unit tests for SpectrumMoleculeModel, loss function, and checkpoint compatibility."""

from pathlib import Path
import torch
from enveda.models import MoleculeEncoder, SpectrumEncoder, SpectrumMoleculeModel
from enveda.training import (
    load_checkpoint,
    multi_positive_contrastive_loss,
    save_checkpoint,
)


def test_model_forward_pass_shapes_and_finite():
    batch_size = 4
    model = SpectrumMoleculeModel(
        spectrum_dim=10000,
        metadata_dim=130,
        fingerprint_dim=2048,
        embedding_dim=128,
    )
    model.eval()

    spectrum = torch.randn(batch_size, 10000)
    metadata = torch.randn(batch_size, 130)
    fingerprint = torch.randint(0, 2, (batch_size, 2048)).float()

    with torch.no_grad():
        spec_emb, mol_emb = model(spectrum, metadata, fingerprint)

    assert spec_emb.shape == (batch_size, 128)
    assert mol_emb.shape == (batch_size, 128)
    assert torch.isfinite(spec_emb).all().item()
    assert torch.isfinite(mol_emb).all().item()

    # Verify L2 normalization
    spec_norms = torch.norm(spec_emb, p=2, dim=1)
    mol_norms = torch.norm(mol_emb, p=2, dim=1)
    assert torch.allclose(spec_norms, torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(mol_norms, torch.ones(batch_size), atol=1e-5)


def test_loss_finite_and_gradients():
    batch_size = 4
    model = SpectrumMoleculeModel()
    model.train()

    spectrum = torch.randn(batch_size, 10000)
    metadata = torch.randn(batch_size, 130)
    fingerprint = torch.randn(batch_size, 2048)
    molecule_ids = torch.tensor([0, 1, 0, 2], dtype=torch.int64)  # 0 has 2 positive spectra

    spec_emb, mol_emb = model(spectrum, metadata, fingerprint)
    loss = multi_positive_contrastive_loss(spec_emb, mol_emb, molecule_ids)

    assert torch.isfinite(loss).item()
    loss.backward()

    # Ensure gradients reached both encoders
    spec_weight_grad = model.spectrum_encoder.network[0].weight.grad
    mol_weight_grad = model.molecule_encoder.network[0].weight.grad

    assert spec_weight_grad is not None and torch.isfinite(spec_weight_grad).all().item()
    assert mol_weight_grad is not None and torch.isfinite(mol_weight_grad).all().item()


def test_checkpoint_save_load_and_resume(tmp_path):
    model = SpectrumMoleculeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_file = tmp_path / "test_ckpt.pt"
    save_checkpoint(
        filepath=ckpt_file,
        model=model,
        optimizer=optimizer,
        step=42,
        loss_history=[0.5, 0.4, 0.3],
    )

    reloaded_model = SpectrumMoleculeModel()
    reloaded_opt = torch.optim.AdamW(reloaded_model.parameters(), lr=1e-4)

    meta = load_checkpoint(ckpt_file, model=reloaded_model, optimizer=reloaded_opt)
    assert meta["step"] == 42
    assert meta["loss_history"] == [0.5, 0.4, 0.3]

    # Verify weights match
    for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.allclose(p1, p2)


def test_existing_checkpoint_compatibility():
    ckpt_path = Path("checkpoints/spectrum_molecule_step_100.pt")
    if not ckpt_path.is_file():
        return

    model = SpectrumMoleculeModel()
    # Strict loading must succeed without missing or unexpected keys
    meta = load_checkpoint(ckpt_path, model=model, device="cpu", strict=True)
    assert meta["step"] == 100
    assert len(meta["loss_history"]) >= 100

    # Test forward pass with loaded weights
    model.eval()
    with torch.no_grad():
        spec = torch.zeros(1, 10000)
        meta_feat = torch.zeros(1, 130)
        fp = torch.zeros(1, 2048)
        s_emb, m_emb = model(spec, meta_feat, fp)
        assert s_emb.shape == (1, 128)
        assert m_emb.shape == (1, 128)
        assert torch.isfinite(s_emb).all().item()
