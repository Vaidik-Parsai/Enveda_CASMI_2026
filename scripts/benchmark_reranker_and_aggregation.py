"""Benchmark script for Aggregation Optimization (Phase 5E) and Candidate Reranking (Phase 5F)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Ensure src is on python path
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
    create_or_load_group_split,
    filter_spectrum,
)
from enveda.evaluation.aggregation import (
    aggregate_by_max_similarity,
    aggregate_by_mean_similarity,
    aggregate_by_rrf,
)
from enveda.evaluation.mass_filter import (
    calculate_exact_monoisotopic_mass,
    compute_candidate_bank_exact_masses,
    compute_soft_mass_scores,
    filter_candidates_by_mass,
    get_molecule_consensus_neutral_mass,
)
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.reranker import CandidateReranker
from enveda.evaluation.retrieval import chunked_topk_search
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def run_reranking_benchmark(checkpoint_path: str = "checkpoints/spectrum_molecule_step_10000.pt"):
    logger = setup_logger("reranker_benchmark")
    logger.info("=" * 70)
    logger.info("PHASE 5E & 5F: AGGREGATION OPTIMIZATION & RERANKING BENCHMARK")
    logger.info("=" * 70)

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")

    # 1. Load Model
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()

    # 2. Candidate Bank & Exact Masses
    ckpt_name = Path(checkpoint_path).stem
    cand_dir = Path(f"outputs/candidate_bank_{ckpt_name}")
    if not cand_dir.is_dir():
        cand_dir = Path("outputs/candidate_bank")
    manifest_file = cand_dir / "candidate_manifest.json"
    emb_file = cand_dir / "candidate_embeddings.npy"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    candidate_smiles = manifest["smiles"]
    candidate_embeddings = np.load(str(emb_file), mmap_mode="r")
    num_candidates = len(candidate_smiles)
    logger.info(f"Loaded candidate bank with {num_candidates:,} molecules.")

    # Candidate masses
    mass_cache = cand_dir / "candidate_exact_masses.npy"
    if not mass_cache.is_file():
        mass_cache = Path("outputs/candidate_bank/candidate_exact_masses.npy")
    candidate_masses = np.load(str(mass_cache))
    logger.info("Loaded candidate exact monoisotopic masses.")

    smiles_to_c_idx = {smi: idx for idx, smi in enumerate(candidate_smiles)}

    # 3. Validation Data (400 molecules)
    preprocessor = MetadataPreprocessor.load(config.data.meta_preprocessor_path)
    train_path = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    _, val_rows = create_or_load_group_split(train_path, split_path)

    val_table = pq.read_table(
        str(train_path),
        columns=[
            "inchikey14",
            "normalized_smiles",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "precursor_mz",
            "base_peak_intensity",
            "collision_energy_ev",
            "adduct",
            "ionization_mode",
        ],
    )
    val_df_full = val_table.to_pandas().iloc[val_rows].reset_index(drop=True)
    mol_grouped = val_df_full.groupby("inchikey14", sort=False)
    unique_inchikeys = list(mol_grouped.groups.keys())[:400]
    val_df = val_df_full[val_df_full["inchikey14"].isin(set(unique_inchikeys))].reset_index(drop=True)
    logger.info(f"Loaded 400 validation molecules ({len(val_df):,} spectra).")

    mol_to_spec_indices = defaultdict(list)
    for s_idx, m_id in enumerate(val_df["inchikey14"]):
        mol_to_spec_indices[m_id].append(s_idx)

    mol_true_cand_idx = {}
    mol_consensus_masses = {}
    for m_id in unique_inchikeys:
        sub_df = val_df[val_df["inchikey14"] == m_id]
        true_smi = sub_df["normalized_smiles"].iloc[0]
        mol_true_cand_idx[m_id] = smiles_to_c_idx[true_smi]
        pmzs = sub_df["precursor_mz"].tolist()
        adds = sub_df["adduct"].tolist()
        modes = sub_df["ionization_mode"].tolist()
        c_mass = get_molecule_consensus_neutral_mass(pmzs, adds, modes)
        if c_mass is None:
            c_mass = calculate_exact_monoisotopic_mass(true_smi)
        mol_consensus_masses[m_id] = c_mass

    # 4. Encode Query Spectra
    logger.info("Encoding validation spectra...")
    meta_features = preprocessor.transform(val_df)
    query_embs_list = []
    batch_size = 64
    with torch.inference_mode():
        for i in range(0, len(val_df), batch_size):
            chunk_df = val_df.iloc[i : i + batch_size]
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
            m_t = torch.from_numpy(meta_features[i : i + len(chunk_df)]).to(device)
            query_embs_list.append(model.encode_spectrum(s_t, m_t).cpu())
    query_embeddings = torch.cat(query_embs_list, dim=0)

    # Convert candidate embeddings to GPU tensor
    cand_embs_tensor = torch.from_numpy(candidate_embeddings[:].copy()).to(device)

    # 5. Define Evaluation Strategies
    strategies = [
        {"name": "Neural RRF (k=30)", "type": "rrf", "k": 30},
        {"name": "Neural RRF (k=60)", "type": "rrf", "k": 60},
        {"name": "Neural RRF (k=100)", "type": "rrf", "k": 100},
        {"name": "Neural Mean Similarity", "type": "mean"},
        {"name": "Neural Max Similarity", "type": "max"},
        {"name": "Mass-Aware Mean (Soft sigma=30)", "type": "mass_mean", "sigma": 30.0},
        {"name": "Mass-Aware RRF (Soft sigma=30)", "type": "mass_rrf", "sigma": 30.0},
        {"name": "Reranker (Top-100 Pool + Multi-Feature)", "type": "reranker", "pool_k": 100},
        {"name": "Reranker (Top-200 Pool + Multi-Feature)", "type": "reranker", "pool_k": 200},
    ]

    reranker = CandidateReranker(
        weight_neural=1.0,
        weight_mass_ppm=2.0,
        sigma_ppm=25.0,
        weight_consistency=0.5,
    )

    benchmark_records = []

    for strat in strategies:
        s_name = strat["name"]
        s_type = strat["type"]
        logger.info(f"Evaluating strategy: {s_name}...")
        ranks = []
        t0 = time.time()

        for m_idx, m_id in enumerate(unique_inchikeys):
            true_c_idx = mol_true_cand_idx[m_id]
            s_indices = mol_to_spec_indices[m_id]
            exp_mass = mol_consensus_masses[m_id]
            q_spectra = query_embeddings[s_indices].to(device)  # shape (num_spec, 128)

            if s_type == "rrf":
                k_val = strat["k"]
                # Per-spectrum cosine similarity against candidates
                spec_sims = (q_spectra @ cand_embs_tensor.T).cpu().numpy()
                rrf_scores = defaultdict(float)
                for s_i in range(len(s_indices)):
                    top_p = np.argsort(-spec_sims[s_i])[:200]
                    for r0, c_i in enumerate(top_p):
                        rrf_scores[c_i] += 1.0 / (k_val + r0 + 1)
                sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:25]
                top25 = [c_i for c_i, _ in sorted_cands]

            elif s_type == "mean":
                q_mean = q_spectra.mean(dim=0)
                sims = (cand_embs_tensor @ q_mean).cpu().numpy()
                top25 = np.argsort(-sims)[:25].tolist()

            elif s_type == "max":
                spec_sims = (q_spectra @ cand_embs_tensor.T).cpu().numpy()
                max_sims = np.max(spec_sims, axis=0)
                top25 = np.argsort(-max_sims)[:25].tolist()

            elif s_type == "mass_mean":
                q_mean = q_spectra.mean(dim=0)
                raw_sims = (cand_embs_tensor @ q_mean).cpu().numpy()
                mass_scores = compute_soft_mass_scores(exp_mass, candidate_masses, sigma_ppm=strat["sigma"])
                comb_sims = (raw_sims + 1.0) * mass_scores
                top25 = np.argsort(-comb_sims)[:25].tolist()

            elif s_type == "mass_rrf":
                spec_sims = (q_spectra @ cand_embs_tensor.T).cpu().numpy()
                mass_scores = compute_soft_mass_scores(exp_mass, candidate_masses, sigma_ppm=strat["sigma"])
                # Apply mass filter weights to each spectrum similarities
                weighted_sims = (spec_sims + 1.0) * mass_scores[None, :]
                rrf_scores = defaultdict(float)
                for s_i in range(len(s_indices)):
                    top_p = np.argsort(-weighted_sims[s_i])[:200]
                    for r0, c_i in enumerate(top_p):
                        rrf_scores[c_i] += 1.0 / (60 + r0 + 1)
                sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:25]
                top25 = [c_i for c_i, _ in sorted_cands]

            elif s_type == "reranker":
                pool_k = strat["pool_k"]
                # 1. First-stage mass-aware retrieval to obtain top pool_k candidates
                q_mean = q_spectra.mean(dim=0)
                raw_sims = (cand_embs_tensor @ q_mean).cpu().numpy()
                mass_scores = compute_soft_mass_scores(exp_mass, candidate_masses, sigma_ppm=30.0)
                first_stage_scores = (raw_sims + 1.0) * mass_scores
                candidate_pool = np.argsort(-first_stage_scores)[:pool_k].tolist()

                # 2. Extract per-spectrum similarities for the candidate pool
                pool_cand_embs = cand_embs_tensor[candidate_pool]  # (pool_k, 128)
                pool_spec_sims = (q_spectra @ pool_cand_embs.T).cpu().numpy()  # (num_spec, pool_k)

                # 3. Rerank using CandidateReranker
                top25, _ = reranker.rerank(
                    candidate_indices=candidate_pool,
                    candidate_masses=candidate_masses,
                    expected_neutral_mass=exp_mass,
                    spectrum_sims=pool_spec_sims,
                    top_k=25,
                )

            # Record rank
            if true_c_idx in top25:
                rank = top25.index(true_c_idx) + 1
            else:
                rank = 26
            ranks.append(rank)

        t_cost = time.time() - t0
        metrics = compute_retrieval_metrics(ranks, top_k_list=[1, 5, 10, 25])
        rec = {
            "Strategy": s_name,
            "MRR@25": metrics["mrr@25"],
            "Recall@1": metrics["recall@1"],
            "Recall@5": metrics["recall@5"],
            "Recall@10": metrics["recall@10"],
            "Recall@25": metrics["recall@25"],
            "Full MRR": metrics["mrr"],
            "Runtime (s)": round(t_cost, 2),
        }
        benchmark_records.append(rec)
        logger.info(
            f"  {s_name:<40} | MRR@25: {metrics['mrr@25']:.4f} | "
            f"R@1: {metrics['recall@1']*100:5.2f}% | R@5: {metrics['recall@5']*100:5.2f}% | "
            f"R@25: {metrics['recall@25']*100:5.2f}% | Cost: {t_cost:.1f}s"
        )

    # 6. Save Table & Report
    df_bm = pd.DataFrame(benchmark_records)
    print("\n" + "=" * 90)
    print("PHASE 5E & 5F: BENCHMARK RESULTS TABLE")
    print("=" * 90)
    print(df_bm.to_string(index=False))

    json_out = Path("outputs/reranker_aggregation_benchmark.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    logger.info(f"Saved benchmark results to: {json_out}")

    return benchmark_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/spectrum_molecule_step_10000.pt")
    args = parser.parse_args()
    run_reranking_benchmark(args.checkpoint)

