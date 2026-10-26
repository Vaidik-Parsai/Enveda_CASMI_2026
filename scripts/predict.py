"""Generate top-25 candidate molecule predictions for test spectra."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
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
    filter_spectrum,
)
from enveda.evaluation import chunked_topk_search
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Top Candidates for Test Spectra")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--candidate_dir", type=str, default="outputs/candidate_bank", help="Path to candidate bank")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test.parquet (optional override)")
    parser.add_argument("--output_csv", type=str, default="outputs/submission_top25.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64, help="Query embedding batch size")
    parser.add_argument("--top_k", type=int, default=25, help="Number of predictions per spectrum")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("enveda_predict")

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
    candidate_embeddings = np.load(str(emb_file), mmap_mode="r")
    logger.info(f"Loaded candidate bank with {len(candidate_embeddings):,} molecules.")

    # Load preprocessor
    meta_path = Path(config.data.meta_preprocessor_path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Preprocessor artifact not found at {meta_path}.")
    preprocessor = MetadataPreprocessor.load(meta_path)

    test_path = Path(args.test_path or config.data.test_path)
    logger.info(f"Reading test spectra from {test_path}...")
    table = pq.read_table(str(test_path))
    test_df = table.to_pandas()
    logger.info(f"Found {len(test_df):,} test spectra.")

    # Process metadata
    meta_feats = preprocessor.transform(test_df)
    query_embs = []

    with torch.inference_mode():
        for i, row in test_df.iterrows():
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

            s_t = torch.from_numpy(spec_vec).unsqueeze(0).to(device)
            m_t = torch.from_numpy(meta_feats[i]).unsqueeze(0).to(device)
            q_emb = model.encode_spectrum(s_t, m_t)
            query_embs.append(q_emb.cpu())

    query_embeddings = torch.cat(query_embs, dim=0)

    logger.info(f"Searching top-{args.top_k} candidates in chunks...")
    topk_indices, _ = chunked_topk_search(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        top_k=args.top_k,
        chunk_size=config.evaluation.candidate_chunk_size,
        device=device,
    )

    # Format predictions
    records = []
    spectrum_ids = test_df.get("spectrum_id", range(len(test_df)))
    for q_idx, spec_id in enumerate(spectrum_ids):
        pred_smiles = [candidate_smiles[idx] for idx in topk_indices[q_idx].tolist()]
        records.append({
            "spectrum_id": spec_id,
            "top_candidates": " ".join(pred_smiles),
        })

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    logger.info(f"Predictions saved to: {out_csv}")


if __name__ == "__main__":
    main()
