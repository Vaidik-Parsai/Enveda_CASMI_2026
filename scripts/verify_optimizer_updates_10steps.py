"""Controlled 10-step optimizer update verification for Phase 6B."""

from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enveda.config import load_config
from enveda.data import (
    MetadataPreprocessor,
    StreamingParquetDataset,
    create_or_load_group_split,
)
from enveda.models import SpectrumMoleculeModel
from enveda.training import (
    CosineWarmupScheduler,
    load_checkpoint,
)
from enveda.training.losses import multi_positive_contrastive_loss


def run_10step_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"CONTROLLED 10-STEP OPTIMIZER UPDATE VERIFICATION (Device: {device})")
    print("=" * 90)

    config = load_config("configs/default.yaml")
    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    train_rows, _ = create_or_load_group_split(train_parquet, split_path)
    preprocessor = MetadataPreprocessor.load(config.data.meta_preprocessor_path)

    # Initialize model from Step 5,000
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)
    load_checkpoint("checkpoints/spectrum_molecule_step_5000.pt", model=model, device=device)

    # Fresh optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=500, total_steps=5000, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)

    # Dataset iterator
    dataset = StreamingParquetDataset(
        parquet_path=str(train_parquet),
        meta_preprocessor=preprocessor,
        batch_size=128,
        row_indices_filter=set(train_rows),
        device=device,
    )
    data_iter = iter(dataset)

    # Key parameter to track actual weight updates
    target_param = model.spectrum_encoder.network[0].weight

    print(f"{'Step':<5} | {'Loss':<8} | {'Grads Finite':<13} | {'Scale':<12} | {'Opt Skipped':<12} | {'Weight Change (L2)':<18} | {'LR Progression':<25}")
    print("-" * 110)

    all_updates_verified = True

    for step_idx in range(10):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = next(data_iter)

        spectrum = batch["spectrum"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        fingerprint = batch["fingerprint"].to(device, non_blocking=True)
        molecule_ids = batch["molecule_ids"].to(device, non_blocking=True)

        weight_before = target_param.detach().clone()

        with torch.amp.autocast("cuda"):
            spec_emb, mol_emb = model(spectrum, metadata, fingerprint)
            loss = multi_positive_contrastive_loss(spec_emb, mol_emb, molecule_ids)

        loss_val = float(loss.item())

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Verify gradient finiteness
        grads_finite = all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        opt_skipped = (scale_before > scale_after)

        # Compute actual weight change L2 norm
        weight_after = target_param.detach().clone()
        weight_delta = torch.norm(weight_after - weight_before).item()

        # Learning rate tracking
        lr_before = optimizer.param_groups[0]["lr"]
        if not opt_skipped:
            scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]

        scale_str = f"{scale_before:.0f}->{scale_after:.0f}"
        lr_str = f"{lr_before:.8f}->{lr_after:.8f}"
        print(f"{step_idx+1:<5d} | {loss_val:<8.4f} | {str(grads_finite):<13} | {scale_str:<12} | {str(opt_skipped):<12} | {weight_delta:<18.6e} | {lr_str:<25}")

        # Strict empirical checks
        if not grads_finite:
            all_updates_verified = False
        if opt_skipped:
            all_updates_verified = False
        if weight_delta <= 0:
            all_updates_verified = False
        if lr_after <= lr_before:
            all_updates_verified = False

    print("=" * 110)
    if all_updates_verified:
        print("[VERIFIED] All 10/10 optimizer steps executed successfully with non-zero parameter changes and proper LR advancement.")
    else:
        print("[FAILED] One or more optimizer steps failed verification.")
    print("=" * 110)


if __name__ == "__main__":
    run_10step_verification()

