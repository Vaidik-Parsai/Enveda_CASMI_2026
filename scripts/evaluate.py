"""Evaluate retrieval performance (Recall@1, 5, 25, MRR) with chunked search."""

import argparse
import json
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
from enveda.data import (
    MetadataPreprocessor,
    bin_spectrum_vector,
    create_or_load_group_split,
    filter_spectrum,
)
from enveda.evaluation import evaluate_retrieval
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower Retrieval Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--candidate_dir", type=str, default="outputs/candidate_bank", help="Path to candidate bank")
    parser.add_argument("--num_queries", type=int, default=1000, help="Number of validation queries to evaluate")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Candidate chunk size for cosine search")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("enveda_evaluate")

    device_str = args.device or config.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    logger.info(f"Loading checkpoint: {args.checkpoint} on {device}")
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)

    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    # Load candidate bank
    cand_path = Path(args.candidate_dir)
    emb_file = cand_path / "candidate_embeddings.npy"
    manifest_file = cand_path / "candidate_manifest.json"

    if not emb_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(
            f"Candidate bank not found at {cand_path}. Run scripts/build_candidate_bank.py first."
        )

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    candidate_smiles = manifest["smiles"]
    smiles_to_candidate_idx = {smi: i for i, smi in enumerate(candidate_smiles)}
    candidate_embeddings = np.load(str(emb_file), mmap_mode="r")
    logger.info(f"Loaded candidate bank with {len(candidate_embeddings):,} candidates.")

    # Load preprocessor
    meta_path = Path(config.data.meta_preprocessor_path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Preprocessor artifact not found at {meta_path}.")
    preprocessor = MetadataPreprocessor.load(meta_path)

    # Load validation split
    train_path = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    _, val_rows = create_or_load_group_split(
        parquet_path=train_path,
        split_cache_path=split_path,
        val_fraction=config.data.val_fraction,
        seed=config.data.seed,
    )

    eval_rows = val_rows[: args.num_queries]
    logger.info(f"Evaluating on {len(eval_rows):,} validation queries...")

    # Read validation records
    table = pq.read_table(
        str(train_path),
        columns=[
            "ms2_mzs",
            "ms2_normalized_intensities",
            "precursor_mz",
            "base_peak_intensity",
            "collision_energy_ev",
            "adduct",
            "ionization_mode",
            "normalized_smiles",
        ],
    )
    val_df = table.to_pandas().iloc[eval_rows].reset_index(drop=True)

    # Encode query spectra
    meta_feats = preprocessor.transform(val_df)
    query_embeddings_list = []
    true_candidate_indices = []

    with torch.inference_mode():
        for i, row in val_df.iterrows():
            mzs, ints = filter_spectrum(
                row.get("ms2_mzs", []),
                row.get("ms2_normalized_intensities", []),
                row.get("precursor_mz", 0.0),
                threshold=config.data.spectrum_threshold,
                top_n=config.data.spectrum_top_n,
            )
            spec_vec = bin_spectrum_vector(
                mzs,
                ints,
                bin_size=config.data.spectrum_bin_size,
                max_mz=config.data.spectrum_max_mz,
                num_bins=config.model.spectrum_dim,
            )

            s_tensor = torch.from_numpy(spec_vec).unsqueeze(0).to(device)
            m_tensor = torch.from_numpy(meta_feats[i]).unsqueeze(0).to(device)

            q_emb = model.encode_spectrum(s_tensor, m_tensor)
            query_embeddings_list.append(q_emb.cpu())

            true_smi = row.get("normalized_smiles", "")
            true_idx = smiles_to_candidate_idx.get(true_smi, -1)
            true_candidate_indices.append(true_idx)

    query_embeddings = torch.cat(query_embeddings_list, dim=0)

    # Run memory-bounded chunked retrieval evaluation
    metrics = evaluate_retrieval(
        query_embeddings=query_embeddings,
        true_candidate_indices=true_candidate_indices,
        candidate_embeddings=candidate_embeddings,
        top_k_list=[1, 5, 25],
        chunk_size=args.chunk_size,
        device=device,
    )

    logger.info("=" * 45)
    logger.info("RETRIEVAL EVALUATION RESULTS")
    logger.info("=" * 45)
    for k, v in metrics.items():
        if "recall" in k or "mrr" in k:
            logger.info(f"{k.upper():12s}: {v:.4f} ({v*100:.2f}%)")
        else:
            logger.info(f"{k.upper():12s}: {v:.2f}")
    logger.info("=" * 45)


if __name__ == "__main__":
    main()
