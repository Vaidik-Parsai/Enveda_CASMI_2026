"""Build disk-backed candidate molecular embedding bank."""

import argparse
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
import torch

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enveda.config import load_config
from enveda.evaluation import build_candidate_bank
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build Candidate Molecular Bank")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/candidate_bank", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=256, help="Candidate embedding batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debug")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("enveda_candidate_bank")

    device_str = args.device or config.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)

    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    train_path = Path(config.data.train_path)
    logger.info(f"Extracting unique candidate SMILES from {train_path}...")
    table = pq.read_table(str(train_path), columns=["normalized_smiles"])
    raw_smiles = table.column(0).to_pandas().dropna().unique().tolist()
    logger.info(f"Found {len(raw_smiles):,} unique candidate SMILES.")

    if args.limit:
        raw_smiles = raw_smiles[: args.limit]
        logger.info(f"Limiting to first {len(raw_smiles):,} candidates for debugging.")

    emb_path, manifest_path = build_candidate_bank(
        smiles_list=raw_smiles,
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        fp_radius=config.data.fingerprint_radius,
        fp_n_bits=config.model.fingerprint_dim,
    )

    logger.info(f"Successfully generated candidate bank:")
    logger.info(f"  Embeddings: {emb_path}")
    logger.info(f"  Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
