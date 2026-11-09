"""Unified benchmark script for Checkpoint Comparison (Phase 5D), Aggregation Optimization (Phase 5E), and Candidate Reranking (Phase 5F)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Tuple
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
    compute_soft_mass_scores,
    get_molecule_consensus_neutral_mass,
)
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.reranker import CandidateReranker
from enveda.evaluation.retrieval import build_candidate_bank
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def get_or_build_candidate_embeddings(
    ckpt_name: str,
    checkpoint_path: Path,
    model: SpectrumMoleculeModel,
    candidate_smiles: List[str],
    device: torch.device,
    config,
    logger,
) -> np.ndarray:
    """Load or encode candidate embeddings matching the exact model checkpoint weights."""
    cand_dir = Path(f"outputs/candidate_bank_{ckpt_name}")
    cand_dir.mkdir(parents=True, exist_ok=True)
    emb_file = cand_dir / "candidate_embeddings.npy"
    man_file = cand_dir / "candidate_manifest.json"

    if emb_file.is_file() and man_file.is_file():
        logger.info(f"Loaded existing candidate embeddings for {ckpt_name} from {emb_file}")
        return np.load(str(emb_file), mmap_mode="r")

    # If step_5000 and base candidate_bank exists, check that
    if ckpt_name == "spectrum_molecule_step_5000" and Path("outputs/candidate_bank/candidate_embeddings.npy").is_file():
        logger.info("Using baseline candidate embeddings from outputs/candidate_bank/")
        return np.load("outputs/candidate_bank/candidate_embeddings.npy", mmap_mode="r")

    logger.info(f"Encoding {len(candidate_smiles):,} candidate molecules with {ckpt_name} on GPU...")
    t0 = time.time()
    build_candidate_bank(
        smiles_list=candidate_smiles,
        model=model,
        output_dir=cand_dir,
        batch_size=512,
        device=device,
        fp_radius=config.data.fingerprint_radius,
        fp_n_bits=config.model.fingerprint_dim,
    )
    logger.info(f"Encoded candidate bank for {ckpt_name} in {time.time() - t0:.1f}s.")
    return np.load(str(emb_file), mmap_mode="r")


def run_comprehensive_benchmark():
    logger = setup_logger("comprehensive_benchmark")
    logger.info("=" * 80)
    logger.info("STARTING PHASE 5D, 5E, 5F COMPREHENSIVE BENCHMARK")
    logger.info("=" * 80)

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Base Manifest and Exact Masses
    base_manifest_path = Path("outputs/candidate_bank/candidate_manifest.json")
    with open(base_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    candidate_smiles = manifest["smiles"]
    num_candidates = len(candidate_smiles)
    logger.info(f"Candidate bank universe: {num_candidates:,} SMILES.")

    candidate_masses = np.load("outputs/candidate_bank/candidate_exact_masses.npy")
    logger.info("Loaded precomputed candidate exact masses.")
    smiles_to_c_idx = {smi: idx for idx, smi in enumerate(candidate_smiles)}

    # 2. Validation Data
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
    unique_inchikeys = list(val_df_full.groupby("inchikey14", sort=False).groups.keys())[:400]
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

    # Pre-parse spectra vectors and metadata features
    logger.info("Extracting validation spectrum feature tensors...")
    meta_features = preprocessor.transform(val_df)
    spec_vectors = []
    for _, row in val_df.iterrows():
        mzs, ints = filter_spectrum(
            row.get("ms2_mzs", []),
            row.get("ms2_normalized_intensities", []),
            row.get("precursor_mz", 0.0),
            threshold=config.data.spectrum_threshold,
            top_n=config.data.spectrum_top_n,
        )
        vec = bin_spectrum_vector(
            mzs,
            ints,
            bin_size=config.data.spectrum_bin_size,
            max_mz=config.data.spectrum_max_mz,
            num_bins=config.model.spectrum_dim,
        )
        spec_vectors.append(vec)
    spec_vectors = np.array(spec_vectors, dtype=np.float32)

    # -------------------------------------------------------------------------
    # PART 1: PHASE 5D — EVALUATE ALL CHECKPOINTS
    # -------------------------------------------------------------------------
    checkpoint_comparison_file = Path("outputs/checkpoint_comparison.json")
    best_archive = Path("checkpoints/best_retrieval_model.pt")

    if checkpoint_comparison_file.is_file() and best_archive.is_file():
        logger.info(f"Loading existing checkpoint comparison from {checkpoint_comparison_file}...")
        with open(checkpoint_comparison_file, "r", encoding="utf-8") as f:
            checkpoint_records = json.load(f)
        df_ckpt = pd.DataFrame(checkpoint_records)
        print("\n" + "=" * 90)
        print("PHASE 5D: CHECKPOINT COMPARISON TABLE (LOADED)")
        print("=" * 90)
        print(df_ckpt[["Step", "Label", "MassScored_MRR@25", "MassScored_Recall@1", "MassScored_Recall@5", "MassScored_Recall@25", "Unconstrained_MRR@25"]].to_string(index=False))
        best_record = max(checkpoint_records, key=lambda x: x["MassScored_MRR@25"])
        best_checkpoint_path = Path(best_record["Checkpoint"])
        best_checkpoint_name = best_record["Label"]
        checkpoint_best_mrr = best_record["MassScored_MRR@25"]
    else:
        checkpoint_list = [
            ("Step 5,000 (Baseline)", "checkpoints/spectrum_molecule_step_5000.pt", 5000),
            ("Step 10,000", "checkpoints/spectrum_molecule_step_10000.pt", 10000),
            ("Step 15,000 (Hard Negatives)", "checkpoints/spectrum_molecule_step_15000.pt", 15000),
            ("Step 20,000 (Hard Negatives)", "checkpoints/spectrum_molecule_step_20000.pt", 20000),
        ]

        checkpoint_records = []
        checkpoint_best_mrr = -1.0
        best_checkpoint_path = None
        best_checkpoint_name = None

        for label, ckpt_path_str, step_num in checkpoint_list:
            ckpt_path = Path(ckpt_path_str)
            if not ckpt_path.is_file():
                logger.warning(f"Checkpoint file {ckpt_path_str} not found. Skipping.")
                continue

            logger.info("-" * 70)
            logger.info(f"Evaluating {label} from {ckpt_path}...")
            logger.info("-" * 70)

            # Initialize model & load weights
            model = SpectrumMoleculeModel(
                spectrum_dim=config.model.spectrum_dim,
                metadata_dim=config.model.metadata_dim,
                fingerprint_dim=config.model.fingerprint_dim,
                embedding_dim=config.model.embedding_dim,
            ).to(device)
            load_checkpoint(ckpt_path, model=model, device=device)
            model.eval()

            # Get or build matching candidate embeddings
            cand_embs_np = get_or_build_candidate_embeddings(
                ckpt_name=ckpt_path.stem,
                checkpoint_path=ckpt_path,
                model=model,
                candidate_smiles=candidate_smiles,
                device=device,
                config=config,
                logger=logger,
            )
            cand_embs_tensor = torch.from_numpy(cand_embs_np[:].copy()).to(device)

            # Encode validation query spectra
            q_embs_list = []
            batch_size = 128
            with torch.inference_mode():
                for i in range(0, len(val_df), batch_size):
                    s_b = torch.from_numpy(spec_vectors[i : i + batch_size]).to(device)
                    m_b = torch.from_numpy(meta_features[i : i + batch_size]).to(device)
                    q_embs_list.append(model.encode_spectrum(s_b, m_b).cpu())
            val_query_embeddings = torch.cat(q_embs_list, dim=0)

            # 1. Unconstrained Mean Retrieval
            ranks_unconstrained = []
            # 2. Soft Mass Mean Retrieval (sigma = 30 ppm)
            ranks_soft_mass = []

            with torch.inference_mode():
                for m_id in unique_inchikeys:
                    true_c_idx = mol_true_cand_idx[m_id]
                    s_indices = mol_to_spec_indices[m_id]
                    exp_mass = mol_consensus_masses[m_id]

                    q_spectra = val_query_embeddings[s_indices].to(device)
                    q_mean = q_spectra.mean(dim=0)
                    sims = (cand_embs_tensor @ q_mean).cpu().numpy()

                    # Unconstrained
                    top25_un = np.argsort(-sims)[:25].tolist()
                    ranks_unconstrained.append(top25_un.index(true_c_idx) + 1 if true_c_idx in top25_un else 26)

                    # Soft mass scoring
                    mass_scores = compute_soft_mass_scores(exp_mass, candidate_masses, sigma_ppm=30.0)
                    comb_sims = (sims + 1.0) * mass_scores
                    top25_sm = np.argsort(-comb_sims)[:25].tolist()
                    ranks_soft_mass.append(top25_sm.index(true_c_idx) + 1 if true_c_idx in top25_sm else 26)

            m_un = compute_retrieval_metrics(ranks_unconstrained, top_k_list=[1, 5, 10, 25])
            m_sm = compute_retrieval_metrics(ranks_soft_mass, top_k_list=[1, 5, 10, 25])

            rec = {
                "Step": step_num,
                "Label": label,
                "Checkpoint": str(ckpt_path),
                "Unconstrained_MRR@25": m_un["mrr@25"],
                "Unconstrained_Recall@25": m_un["recall@25"],
                "MassScored_MRR@25": m_sm["mrr@25"],
                "MassScored_Recall@1": m_sm["recall@1"],
                "MassScored_Recall@5": m_sm["recall@5"],
                "MassScored_Recall@10": m_sm["recall@10"],
                "MassScored_Recall@25": m_sm["recall@25"],
                "MassScored_Full_MRR": m_sm["mrr"],
            }
            checkpoint_records.append(rec)
            logger.info(
                f"Step {step_num:5d} | MassScored MRR@25: {m_sm['mrr@25']:.4f} | "
                f"R@1: {m_sm['recall@1']*100:5.2f}% | R@5: {m_sm['recall@5']*100:5.2f}% | "
                f"R@25: {m_sm['recall@25']*100:5.2f}% | Unconstrained MRR@25: {m_un['mrr@25']:.4f}"
            )

            if m_sm["mrr@25"] > checkpoint_best_mrr:
                checkpoint_best_mrr = m_sm["mrr@25"]
                best_checkpoint_path = ckpt_path
                best_checkpoint_name = label

        # Save checkpoint comparison
        df_ckpt = pd.DataFrame(checkpoint_records)
        print("\n" + "=" * 90)
        print("PHASE 5D: CHECKPOINT COMPARISON TABLE")
        print("=" * 90)
        print(df_ckpt[["Step", "Label", "MassScored_MRR@25", "MassScored_Recall@1", "MassScored_Recall@5", "MassScored_Recall@25", "Unconstrained_MRR@25"]].to_string(index=False))

        with open("outputs/checkpoint_comparison.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint_records, f, indent=2)

        logger.info(f"Best performing checkpoint: {best_checkpoint_name} with MRR@25 = {checkpoint_best_mrr:.4f}")
        shutil.copy2(best_checkpoint_path, best_archive)
        logger.info(f"Archived best checkpoint to: {best_archive}")

    # -------------------------------------------------------------------------
    # PART 2: PHASE 5E & 5F — AGGREGATION & RERANKING ON BEST MODEL
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"RUNNING AGGREGATION & RERANKING OPTIMIZATION ON BEST MODEL ({best_checkpoint_name})")
    logger.info("=" * 80)

    # Load best model and embeddings
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)
    load_checkpoint(best_archive, model=model, device=device)
    model.eval()

    cand_embs_np = get_or_build_candidate_embeddings(
        ckpt_name=best_checkpoint_path.stem,
        checkpoint_path=best_checkpoint_path,
        model=model,
        candidate_smiles=candidate_smiles,
        device=device,
        config=config,
        logger=logger,
    )
    cand_embs_tensor = torch.from_numpy(cand_embs_np[:].copy()).to(device)

    # Encode validation queries
    q_embs_list = []
    with torch.inference_mode():
        for i in range(0, len(val_df), 128):
            s_b = torch.from_numpy(spec_vectors[i : i + 128]).to(device)
            m_b = torch.from_numpy(meta_features[i : i + 128]).to(device)
            q_embs_list.append(model.encode_spectrum(s_b, m_b).cpu())
    val_query_embeddings = torch.cat(q_embs_list, dim=0)

    cand_masses_tensor = torch.from_numpy(candidate_masses).to(device)

    strategies = [
        {"name": "Neural Max Similarity", "type": "max"},
        {"name": "Neural Mean Similarity", "type": "mean"},
        {"name": "Neural RRF (k=30)", "type": "rrf", "k": 30},
        {"name": "Neural RRF (k=60)", "type": "rrf", "k": 60},
        {"name": "Neural RRF (k=100)", "type": "rrf", "k": 100},
        {"name": "Mass-Aware Mean (sigma=30 ppm)", "type": "mass_mean", "sigma": 30.0},
        {"name": "Mass-Aware Mean (sigma=20 ppm)", "type": "mass_mean", "sigma": 20.0},
        {"name": "Mass-Aware RRF (sigma=30 ppm, k=60)", "type": "mass_rrf", "sigma": 30.0, "k": 60},
        {"name": "Reranker (Top-100 Pool + Multi-Feature)", "type": "reranker", "pool_k": 100},
        {"name": "Reranker (Top-200 Pool + Multi-Feature)", "type": "reranker", "pool_k": 200},
    ]

    reranker = CandidateReranker(
        weight_neural=1.0,
        weight_mass_ppm=2.0,
        sigma_ppm=25.0,
        weight_consistency=0.5,
    )

    strategy_records = []
    with torch.inference_mode():
        for strat in strategies:
            s_name = strat["name"]
            s_type = strat["type"]
            ranks = []
            t0 = time.time()

            for m_id in unique_inchikeys:
                true_c_idx = mol_true_cand_idx[m_id]
                s_indices = mol_to_spec_indices[m_id]
                exp_mass = mol_consensus_masses[m_id]
                q_spectra = val_query_embeddings[s_indices].to(device)

                if s_type == "max":
                    sims = q_spectra @ cand_embs_tensor.T
                    max_sims = sims.max(dim=0).values
                    top25 = torch.topk(max_sims, k=25).indices.cpu().tolist()

                elif s_type == "mean":
                    q_mean = q_spectra.mean(dim=0)
                    sims = cand_embs_tensor @ q_mean
                    top25 = torch.topk(sims, k=25).indices.cpu().tolist()

                elif s_type == "rrf":
                    k_val = strat["k"]
                    sims = q_spectra @ cand_embs_tensor.T
                    top_indices = torch.topk(sims, k=200, dim=1).indices.cpu().numpy()
                    rrf_scores = defaultdict(float)
                    for s_i in range(len(s_indices)):
                        for r0, c_i in enumerate(top_indices[s_i]):
                            rrf_scores[int(c_i)] += 1.0 / (k_val + r0 + 1)
                    sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:25]
                    top25 = [c_i for c_i, _ in sorted_cands]

                elif s_type == "mass_mean":
                    q_mean = q_spectra.mean(dim=0)
                    sims = cand_embs_tensor @ q_mean
                    delta_m = torch.abs(cand_masses_tensor - exp_mass)
                    ppm = (delta_m / exp_mass) * 1e6
                    m_scores = torch.exp(-0.5 * (ppm / strat["sigma"]) ** 2)
                    comb_sims = (sims + 1.0) * m_scores
                    top25 = torch.topk(comb_sims, k=25).indices.cpu().tolist()

                elif s_type == "mass_rrf":
                    k_val = strat["k"]
                    sims = q_spectra @ cand_embs_tensor.T
                    delta_m = torch.abs(cand_masses_tensor - exp_mass)
                    ppm = (delta_m / exp_mass) * 1e6
                    m_scores = torch.exp(-0.5 * (ppm / strat["sigma"]) ** 2)
                    weighted_sims = (sims + 1.0) * m_scores.unsqueeze(0)
                    top_indices = torch.topk(weighted_sims, k=200, dim=1).indices.cpu().numpy()
                    rrf_scores = defaultdict(float)
                    for s_i in range(len(s_indices)):
                        for r0, c_i in enumerate(top_indices[s_i]):
                            rrf_scores[int(c_i)] += 1.0 / (k_val + r0 + 1)
                    sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:25]
                    top25 = [c_i for c_i, _ in sorted_cands]

                elif s_type == "reranker":
                    pool_k = strat["pool_k"]
                    q_mean = q_spectra.mean(dim=0)
                    sims = cand_embs_tensor @ q_mean
                    delta_m = torch.abs(cand_masses_tensor - exp_mass)
                    ppm = (delta_m / exp_mass) * 1e6
                    m_scores = torch.exp(-0.5 * (ppm / 30.0) ** 2)
                    comb_sims = (sims + 1.0) * m_scores
                    candidate_pool = torch.topk(comb_sims, k=pool_k).indices.cpu().tolist()

                    pool_cand_embs = cand_embs_tensor[candidate_pool]
                    pool_spec_sims = (q_spectra @ pool_cand_embs.T).cpu().numpy()

                    top25, _ = reranker.rerank(
                        candidate_indices=candidate_pool,
                        candidate_masses=candidate_masses,
                        expected_neutral_mass=exp_mass,
                        spectrum_sims=pool_spec_sims,
                        top_k=25,
                    )

                rank = top25.index(true_c_idx) + 1 if true_c_idx in top25 else 26
                ranks.append(rank)

            t_cost = time.time() - t0
            m = compute_retrieval_metrics(ranks, top_k_list=[1, 5, 10, 25])
            rec = {
                "Strategy": s_name,
                "MRR@25": m["mrr@25"],
                "Recall@1": m["recall@1"],
                "Recall@5": m["recall@5"],
                "Recall@10": m["recall@10"],
                "Recall@25": m["recall@25"],
                "Full MRR": m["mrr"],
                "Runtime (s)": round(t_cost, 2),
            }
            strategy_records.append(rec)
        logger.info(
            f"  {s_name:<42} | MRR@25: {m['mrr@25']:.4f} | "
            f"R@1: {m['recall@1']*100:5.2f}% | R@5: {m['recall@5']*100:5.2f}% | "
            f"R@25: {m['recall@25']*100:5.2f}% | Cost: {t_cost:.1f}s"
        )

    df_strat = pd.DataFrame(strategy_records)
    print("\n" + "=" * 90)
    print("PHASE 5E & 5F: STRATEGY BENCHMARK RESULTS")
    print("=" * 90)
    print(df_strat.to_string(index=False))

    with open("outputs/reranker_aggregation_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(strategy_records, f, indent=2)

    logger.info("Saved outputs/reranker_aggregation_benchmark.json")
    logger.info("Comprehensive benchmark complete.")


if __name__ == "__main__":
    run_comprehensive_benchmark()

