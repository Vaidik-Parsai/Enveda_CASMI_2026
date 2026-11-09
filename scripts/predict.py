"""Generate top-25 candidate molecule predictions per molecule for test spectra.

Groups test spectra by `molecule_id` and aggregates predictions using
Mass-Aware Mean, Candidate Reranking, Mean, Max, or RRF.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
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
    MetadataPreprocessor,
    bin_spectrum_vector,
    filter_spectrum,
)
from enveda.evaluation.aggregation import (
    aggregate_by_max_similarity,
    aggregate_by_mean_similarity,
    aggregate_by_rrf,
)
from enveda.evaluation.mass_filter import (
    compute_soft_mass_scores,
    get_molecule_consensus_neutral_mass,
)
from enveda.evaluation.reranker import CandidateReranker
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Top Candidates for Test Molecules")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_retrieval_model.pt", help="Checkpoint path")
    parser.add_argument("--candidate_dir", type=str, default="outputs/candidate_bank", help="Path to candidate bank")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test.parquet (optional override)")
    parser.add_argument("--output_csv", type=str, default="outputs/submission_top25.csv", help="Output CSV path")
    parser.add_argument("--method", type=str, default="mass_mean", choices=["mass_mean", "rerank", "mean", "rrf", "max"], help="Aggregation method")
    parser.add_argument("--top_k", type=int, default=25, help="Number of predictions per molecule")
    parser.add_argument("--sigma_ppm", type=float, default=20.0, help="Gaussian ppm tolerance for mass scoring")
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

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = Path("checkpoints/spectrum_molecule_step_5000.pt")

    logger.info(f"Loading checkpoint: {ckpt_path} on {device}")
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)

    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    # Load candidate bank
    cand_path = Path(args.candidate_dir)
    emb_file = cand_path / "candidate_embeddings.npy"
    manifest_file = cand_path / "candidate_manifest.json"
    mass_file = cand_path / "candidate_exact_masses.npy"

    if not emb_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"Candidate bank not found at {cand_path}.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    candidate_smiles = manifest["smiles"]
    candidate_embeddings = np.load(str(emb_file), mmap_mode="r")
    num_candidates = len(candidate_smiles)
    logger.info(f"Loaded candidate bank with {num_candidates:,} molecules.")

    candidate_masses = None
    if mass_file.is_file():
        candidate_masses = np.load(str(mass_file))
        logger.info("Loaded precomputed candidate exact masses.")
    else:
        logger.warning("Candidate exact masses not found; computing...")
        from enveda.evaluation.mass_filter import compute_candidate_bank_exact_masses
        candidate_masses = compute_candidate_bank_exact_masses(candidate_smiles, mass_file)

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

    # Group test spectra by molecule_id
    if "molecule_id" not in test_df.columns:
        raise KeyError(f"'molecule_id' column not found in {test_path}.")

    unique_molecule_ids = []
    mol_to_indices = defaultdict(list)
    for s_idx, m_id in enumerate(test_df["molecule_id"]):
        if m_id not in mol_to_indices:
            unique_molecule_ids.append(m_id)
        mol_to_indices[m_id].append(s_idx)

    num_test_molecules = len(unique_molecule_ids)
    logger.info(f"Identified {num_test_molecules:,} unique test molecules to predict.")

    # Process metadata & encode spectra
    meta_feats = preprocessor.transform(test_df)
    query_embs = []

    batch_size = 128
    with torch.inference_mode():
        for i in range(0, len(test_df), batch_size):
            chunk_df = test_df.iloc[i : i + batch_size]
            spec_batch = []
            for _, row in chunk_df.iterrows():
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
                spec_batch.append(spec_vec)

            s_t = torch.from_numpy(np.array(spec_batch, dtype=np.float32)).to(device)
            m_t = torch.from_numpy(meta_feats[i : i + len(chunk_df)]).to(device)
            q_emb = model.encode_spectrum(s_t, m_t)
            query_embs.append(q_emb.cpu())

    query_embeddings = torch.cat(query_embs, dim=0)
    logger.info(f"Encoded {len(query_embeddings):,} test spectrum embeddings.")

    cand_embs_tensor = torch.from_numpy(candidate_embeddings[:].copy()).to(device)
    cand_masses_tensor = torch.from_numpy(candidate_masses).to(device)

    logger.info(f"Generating predictions using strategy: {args.method.upper()}...")
    reranker = CandidateReranker(
        weight_neural=1.0,
        weight_mass_ppm=2.0,
        sigma_ppm=25.0,
        weight_consistency=0.5,
    )

    records = []
    with torch.inference_mode():
        for m_id in unique_molecule_ids:
            s_indices = mol_to_indices[m_id]
            q_spectra = query_embeddings[s_indices].to(device)
            q_mean = q_spectra.mean(dim=0)

            # Consensus neutral mass estimation
            sub_df = test_df.iloc[s_indices]
            pmzs = sub_df["precursor_mz"].tolist()
            adds = sub_df["adduct"].tolist()
            modes = sub_df["ionization_mode"].tolist()
            consensus_mass = get_molecule_consensus_neutral_mass(pmzs, adds, modes)

            if args.method == "mass_mean" and consensus_mass is not None:
                sims = cand_embs_tensor @ q_mean
                delta_m = torch.abs(cand_masses_tensor - consensus_mass)
                ppm = (delta_m / consensus_mass) * 1e6
                m_scores = torch.exp(-0.5 * (ppm / args.sigma_ppm) ** 2)
                comb_sims = (sims + 1.0) * m_scores
                top_indices = torch.topk(comb_sims, k=args.top_k).indices.cpu().tolist()

            elif args.method == "rerank" and consensus_mass is not None:
                sims = cand_embs_tensor @ q_mean
                delta_m = torch.abs(cand_masses_tensor - consensus_mass)
                ppm = (delta_m / consensus_mass) * 1e6
                m_scores = torch.exp(-0.5 * (ppm / 30.0) ** 2)
                comb_sims = (sims + 1.0) * m_scores
                cand_pool = torch.topk(comb_sims, k=200).indices.cpu().tolist()

                pool_cand_embs = cand_embs_tensor[cand_pool]
                pool_spec_sims = (q_spectra @ pool_cand_embs.T).cpu().numpy()

                top_indices, _ = reranker.rerank(
                    candidate_indices=cand_pool,
                    candidate_masses=candidate_masses,
                    expected_neutral_mass=consensus_mass,
                    spectrum_sims=pool_spec_sims,
                    top_k=args.top_k,
                )

            elif args.method == "mean" or consensus_mass is None:
                sims = cand_embs_tensor @ q_mean
                top_indices = torch.topk(sims, k=args.top_k).indices.cpu().tolist()

            elif args.method == "max":
                sims = q_spectra @ cand_embs_tensor.T
                max_sims = sims.max(dim=0).values
                top_indices = torch.topk(max_sims, k=args.top_k).indices.cpu().tolist()

            elif args.method == "rrf":
                sims = q_spectra @ cand_embs_tensor.T
                top200 = torch.topk(sims, k=200, dim=1).indices.cpu().numpy()
                rrf_scores = defaultdict(float)
                for s_i in range(len(s_indices)):
                    for r0, c_i in enumerate(top200[s_i]):
                        rrf_scores[int(c_i)] += 1.0 / (60 + r0 + 1)
                sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
                top_indices = [c_i for c_i, _ in sorted_cands]

            pred_smiles = [candidate_smiles[idx] for idx in top_indices]

            # Format space-delimited string
            records.append({
                "molecule_id": m_id,
                "top_candidates": " ".join(pred_smiles),
            })

    out_df = pd.DataFrame(records)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Strict submission integrity verification
    assert len(out_df) == num_test_molecules, f"Row count {len(out_df)} != expected {num_test_molecules}"
    assert out_df["molecule_id"].nunique() == num_test_molecules, "Duplicate molecule IDs detected!"
    assert out_df["top_candidates"].isna().sum() == 0, "Missing predictions detected!"

    # Verify each row has exactly 25 SMILES
    for idx, row in out_df.iterrows():
        preds = row["top_candidates"].split()
        assert len(preds) == args.top_k, f"Row {idx} has {len(preds)} candidates, expected {args.top_k}"

    logger.info("=" * 70)
    logger.info("SUBMISSION INTEGRITY VERIFICATION SUCCESSFUL")
    logger.info("=" * 70)
    logger.info(f"  Rows (Test Molecules): {len(out_df)}")
    logger.info(f"  Candidates per row:    {args.top_k}")
    logger.info(f"  Output saved to:       {out_csv}")

    # Also save as submission_test.csv for convenience
    copy_path = Path("outputs/submission_test.csv")
    out_df.to_csv(copy_path, index=False)
    logger.info(f"  Mirrored to:           {copy_path}")


if __name__ == "__main__":
    main()
