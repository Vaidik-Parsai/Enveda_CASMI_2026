"""Phase 6B: Chemistry-Aware Contrastive Training Runner.

Executes four controlled experimental runs initialized from Step 5,000 weights:
  - Run A: Baseline MIL-NCE + Cosine LR Schedule
  - Run B: Normalized SupCon + Cosine LR Schedule
  - Run C: Baseline MIL-NCE + Cosine LR Schedule + Chemistry-Aware Negatives
  - Run D: Normalized SupCon + Cosine LR Schedule + Chemistry-Aware Negatives

All runs use an experiment-local step counter (0 to 5,000 steps) with 500-step linear warmup
and cosine annealing to 1e-6. Reference checkpoint (step 5,000) is strictly protected.
"""

import argparse
from pathlib import Path
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
from enveda.data.preprocessing import generate_morgan_fingerprint
from enveda.models import SpectrumMoleculeModel
from enveda.training import (
    CosineWarmupScheduler,
    Trainer,
    load_checkpoint,
    save_checkpoint,
)
from enveda.utils import set_seed, setup_logger


RUN_CONFIGS = {
    "run_a": {
        "name": "Run A",
        "description": "Baseline MIL-NCE + Cosine LR",
        "loss_fn_type": "multi_positive",
        "use_negative_pool": False,
        "checkpoint_dir": "checkpoints/phase6b_run_a",
    },
    "run_b": {
        "name": "Run B",
        "description": "Normalized SupCon + Cosine LR",
        "loss_fn_type": "normalized_supcon",
        "use_negative_pool": False,
        "checkpoint_dir": "checkpoints/phase6b_run_b",
    },
    "run_c": {
        "name": "Run C",
        "description": "Baseline MIL-NCE + Cosine LR + Chemistry Negatives",
        "loss_fn_type": "multi_positive",
        "use_negative_pool": True,
        "checkpoint_dir": "checkpoints/phase6b_run_c",
    },
    "run_d": {
        "name": "Run D",
        "description": "Normalized SupCon + Cosine LR + Chemistry Negatives",
        "loss_fn_type": "normalized_supcon",
        "use_negative_pool": True,
        "checkpoint_dir": "checkpoints/phase6b_run_d",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6B Chemistry-Aware Training Runner")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        choices=["run_a", "run_b", "run_c", "run_d"],
        help="Experiment identifier (run_a, run_b, run_c, run_d)",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="checkpoints/spectrum_molecule_step_5000.pt",
        help="Initial model checkpoint to branch from (default: step 5,000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Base configuration YAML path",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=5000,
        help="Total experiment-local training steps (default: 5,000)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Linear warmup steps (default: 500)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate floor at cycle end (default: 1e-6)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Steps between periodic checkpoint saves (default: 1,000)",
    )
    parser.add_argument(
        "--negatives_per_anchor",
        type=int,
        default=2,
        help="Number of external chemistry-aware negatives per anchor (default: 2)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing run checkpoint to resume from if interrupted",
    )
    return parser.parse_args()


def build_training_negative_pool(
    train_parquet: Path,
    train_rows: list,
    logger,
    max_molecules: int = 150000,
) -> ChemistryAwareNegativePool:
    """Build a ChemistryAwareNegativePool from the training split molecules."""
    logger.info(f"Extracting training molecules for ChemistryAwareNegativePool (sample limit: {max_molecules:,})...")
    table = pq.read_table(
        str(train_parquet),
        columns=["normalized_smiles", "inchikey14", "precursor_mz", "adduct", "ionization_mode"],
    )
    # Filter strictly to training rows
    sample_indices = train_rows[:max_molecules] if len(train_rows) > max_molecules else train_rows
    train_df = table.to_pandas().iloc[sample_indices].reset_index(drop=True)
    unique_mols = train_df.drop_duplicates(subset=["inchikey14"]).reset_index(drop=True)

    smiles_list = unique_mols["normalized_smiles"].tolist()
    inchikeys_list = unique_mols["inchikey14"].tolist()

    logger.info(f"Generating Morgan fingerprints for {len(smiles_list):,} unique training molecules...")
    fps_list = []
    valid_smiles = []
    valid_inchikeys = []

    for s, ikey in zip(smiles_list, inchikeys_list):
        fp_vec, ok = generate_morgan_fingerprint(s, radius=2, n_bits=2048)
        if ok:
            fps_list.append(fp_vec)
            valid_smiles.append(s)
            valid_inchikeys.append(ikey)

    fps_arr = np.array(fps_list, dtype=np.uint8)
    pool = ChemistryAwareNegativePool(
        training_smiles=valid_smiles,
        training_inchikeys=valid_inchikeys,
        training_fingerprints=fps_arr,
        mass_ppm_tolerance=50.0,
        random_seed=42,
    )
    logger.info(f"Initialized ChemistryAwareNegativePool with {pool.num_molecules:,} training molecules.")
    return pool


def main():
    args = parse_args()
    run_info = RUN_CONFIGS[args.run_id]
    config = load_config(args.config)

    ckpt_dir = Path(run_info["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        f"phase6b_{args.run_id}",
        log_file=str(ckpt_dir / f"{args.run_id}.log"),
    )

    logger.info("=" * 80)
    logger.info(f"PHASE 6B: {run_info['name'].upper()} - {run_info['description']}")
    logger.info("=" * 80)
    logger.info(f"Run ID:              {args.run_id}")
    logger.info(f"Objective Function:  {run_info['loss_fn_type']}")
    logger.info(f"Negative Sampling:   {run_info['use_negative_pool']}")
    logger.info(f"Initial Checkpoint:  {args.init_checkpoint}")
    logger.info(f"Output Directory:    {ckpt_dir}")
    logger.info(f"Total Local Steps:   {args.total_steps}")
    logger.info(f"Warmup Steps:        {args.warmup_steps}")
    logger.info(f"Min Learning Rate:   {args.min_lr}")

    set_seed(config.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device:      {device} (GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'})")

    # 1. Dataset split
    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    train_rows, val_rows = create_or_load_group_split(
        parquet_path=train_parquet,
        split_cache_path=split_path,
        val_fraction=config.data.val_fraction,
        seed=config.data.seed,
    )
    logger.info(f"Loaded split: {len(train_rows):,} train spectra | {len(val_rows):,} val spectra")

    # 2. Metadata Preprocessor
    preprocessor = MetadataPreprocessor.load(config.data.meta_preprocessor_path)

    # 3. Model Architecture
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
        dropout_1=config.model.dropout_1,
        dropout_2=config.model.dropout_2,
    ).to(device)

    # Load Step 5,000 weights (or resumed checkpoint)
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        load_checkpoint(args.resume, model=model, device=device)
    else:
        logger.info(f"Branching from reference weights: {args.init_checkpoint}")
        load_checkpoint(args.init_checkpoint, model=model, device=device)

    # 4. Fresh Optimizer & Cosine Warmup Scheduler
    base_lr = config.training.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        min_lr=args.min_lr,
    )

    # 5. Negative Pool (if enabled for Runs C and D)
    negative_pool = None
    if run_info["use_negative_pool"]:
        negative_pool = build_training_negative_pool(
            train_parquet=train_parquet,
            train_rows=train_rows,
            logger=logger,
        )

    # 6. Training Dataset
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

    # Configure Trainer
    train_config = dict(config)
    train_config["training"]["checkpoint_dir"] = str(ckpt_dir)
    train_config["training"]["checkpoint_interval"] = args.checkpoint_interval
    train_config["training"]["max_steps"] = args.total_steps
    train_config["training"]["scaler_init_scale"] = 8192.0

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        negative_pool=negative_pool,
        loss_fn_type=run_info["loss_fn_type"],
        negatives_per_anchor=args.negatives_per_anchor if run_info["use_negative_pool"] else 0,
        config=train_config,
        device=device,
        train_dataset=train_dataset,
        logger=logger,
    )

    start_step = 0
    if args.resume:
        logger.info(f"Restoring optimizer, scaler, and scheduler state from {args.resume}...")
        ckpt_meta = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=trainer.scaler,
            scheduler=scheduler,
            device=device,
        )
        start_step = ckpt_meta.get("step", 0)
        trainer.loss_history = list(ckpt_meta.get("loss_history", []))
        logger.info(f"Resumed from step {start_step} (loss history: {len(trainer.loss_history)} entries)")

    logger.info(f"Starting {run_info['name']} from step {start_step + 1} to {args.total_steps}...")
    loss_history = trainer.train(start_step=start_step)

    # Final checkpoint save
    final_ckpt = ckpt_dir / f"step_{args.total_steps}.pt"
    save_checkpoint(
        filepath=final_ckpt,
        model=model,
        optimizer=optimizer,
        scaler=trainer.scaler,
        scheduler=scheduler,
        step=args.total_steps,
        loss_history=loss_history,
        config=train_config,
    )
    logger.info(f"Final {run_info['name']} checkpoint saved to: {final_ckpt}")
    logger.info(f"{run_info['name']} completed successfully.")


if __name__ == "__main__":
    main()

