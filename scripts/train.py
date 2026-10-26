"""Training CLI entry point for Enveda CASMI retrieval model."""

import argparse
from pathlib import Path
import sys
import torch

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enveda.config import load_config
from enveda.data import (
    MetadataPreprocessor,
    StreamingParquetDataset,
    create_or_load_group_split,
)
from enveda.models import SpectrumMoleculeModel
from enveda.training import Trainer, load_checkpoint
from enveda.utils import set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Enveda CASMI Two-Tower Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Overrides
    if args.steps is not None:
        config.training.max_steps = args.steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    device_str = args.device or config.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    logger = setup_logger("enveda", log_file=str(Path(config.training.output_dir) / "train.log"))
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Compute device: {device} (GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'})")

    set_seed(config.data.seed)

    # 1. Split
    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)

    logger.info("Initializing train/validation split...")
    train_rows, val_rows = create_or_load_group_split(
        parquet_path=train_parquet,
        split_cache_path=split_path,
        val_fraction=config.data.val_fraction,
        seed=config.data.seed,
    )
    logger.info(f"Train split: {len(train_rows):,} spectra | Val split: {len(val_rows):,} spectra")

    # 2. Preprocessor
    meta_path = Path(config.data.meta_preprocessor_path)
    if meta_path.is_file():
        logger.info(f"Loading metadata preprocessor from {meta_path}")
        preprocessor = MetadataPreprocessor.load(meta_path)
    else:
        logger.info("Fitting metadata preprocessor on training partition...")
        import pyarrow.parquet as pq

        table = pq.read_table(
            str(train_parquet),
            columns=[
                "collision_energy_ev",
                "precursor_mz",
                "base_peak_intensity",
                "adduct",
                "ionization_mode",
            ],
        )
        train_df = table.to_pandas().iloc[train_rows].reset_index(drop=True)
        preprocessor = MetadataPreprocessor(target_dim=config.model.metadata_dim)
        preprocessor.fit(train_df)
        preprocessor.save(meta_path)
        logger.info(f"Saved fitted metadata preprocessor to {meta_path}")
        del train_df, table

    # 3. Model & Optimizer
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
        dropout_1=config.model.dropout_1,
        dropout_2=config.model.dropout_2,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized SpectrumMoleculeModel with {trainable_params:,} trainable parameters.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    start_step = 0
    if args.resume:
        logger.info(f"Resuming model and optimizer from checkpoint: {args.resume}")
        ckpt_meta = load_checkpoint(args.resume, model=model, optimizer=optimizer, device=device)
        start_step = ckpt_meta.get("step", 0)
        logger.info(f"Resumed from step {start_step}")

    # 4. Streaming Dataset
    train_row_set = set(train_rows)
    train_dataset = StreamingParquetDataset(
        parquet_path=str(train_parquet),
        meta_preprocessor=preprocessor,
        batch_size=config.training.batch_size,
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

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        train_dataset=train_dataset,
        logger=logger,
    )

    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()
