"""Controlled validation audit: Mass-scoring ablation (Section A) and Step 5k vs 10k comparison (Section D)."""

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
from enveda.evaluation.mass_filter import (
    calculate_exact_monoisotopic_mass,
    compute_soft_mass_scores,
    get_molecule_consensus_neutral_mass,
)
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import setup_logger


def run_ablation():
    logger = setup_logger("mass_ablation")
    logger.info("=" * 80)
    logger.info("SECTION A & D: CONTROLLED VALIDATION AUDIT (MASS ABLATION & 5K VS 10K)")
    logger.info("=" * 80)

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    # 1. Load candidate manifest and precomputed exact masses
    base_manifest_path = Path("outputs/candidate_bank/candidate_manifest.json")
    with open(base_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    candidate_smiles = manifest["smiles"]
    num_candidates = len(candidate_smiles)
    logger.info(f"Loaded candidate universe: {num_candidates:,} SMILES.")

    candidate_masses = np.load("outputs/candidate_bank/candidate_exact_masses.npy")
    cand_masses_tensor = torch.from_numpy(candidate_masses).to(device)
    logger.info("Loaded candidate exact masses.")
    smiles_to_c_idx = {smi: idx for idx, smi in enumerate(candidate_smiles)}

    # 2. Load identical 400 validation molecules
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

    # Pre-parse validation spectra
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

    # Checkpoints to evaluate
    checkpoints = [
        ("Step 5,000", "checkpoints/spectrum_molecule_step_5000.pt", "outputs/candidate_bank/candidate_embeddings.npy"),
        ("Step 10,000", "checkpoints/spectrum_molecule_step_10000.pt", "outputs/candidate_bank_spectrum_molecule_step_10000/candidate_embeddings.npy"),
    ]

    all_ablation_results = []

    for ckpt_label, ckpt_path_str, cand_emb_path_str in checkpoints:
        logger.info("=" * 70)
        logger.info(f"AUDITING CHECKPOINT: {ckpt_label} ({ckpt_path_str})")
        logger.info("=" * 70)

        ckpt_path = Path(ckpt_path_str)
        cand_emb_path = Path(cand_emb_path_str)

        # Load model
        model = SpectrumMoleculeModel(
            spectrum_dim=config.model.spectrum_dim,
            metadata_dim=config.model.metadata_dim,
            fingerprint_dim=config.model.fingerprint_dim,
            embedding_dim=config.model.embedding_dim,
        ).to(device)
        load_checkpoint(ckpt_path, model=model, device=device)
        model.eval()

        # Load candidate embeddings matching checkpoint
        cand_embs_np = np.load(str(cand_emb_path), mmap_mode="r")
        cand_embs_tensor = torch.from_numpy(cand_embs_np[:].copy()).to(device)

        # Encode validation spectra
        q_embs_list = []
        batch_size = 128
        with torch.inference_mode():
            for i in range(0, len(val_df), batch_size):
                s_b = torch.from_numpy(spec_vectors[i : i + batch_size]).to(device)
                m_b = torch.from_numpy(meta_features[i : i + batch_size]).to(device)
                q_embs_list.append(model.encode_spectrum(s_b, m_b).cpu())
        val_query_embeddings = torch.cat(q_embs_list, dim=0)

        # Conditions to evaluate
        conditions = [
            {"name": "1. Neural similarity only", "use_neural": True, "use_mass": False, "sigma": None},
            {"name": "2. Mass compatibility only", "use_neural": False, "use_mass": True, "sigma": 30.0},
            {"name": "3. Neural + mass (sigma 30 ppm)", "use_neural": True, "use_mass": True, "sigma": 30.0},
            {"name": "4. Neural + mass (sigma 50 ppm)", "use_neural": True, "use_mass": True, "sigma": 50.0},
            {"name": "5. Neural + mass (sigma 20 ppm)", "use_neural": True, "use_mass": True, "sigma": 20.0},
        ]

        with torch.inference_mode():
            for cond in conditions:
                cond_name = cond["name"]
                ranks = []
                t0 = time.time()

                for m_id in unique_inchikeys:
                    true_c_idx = mol_true_cand_idx[m_id]
                    s_indices = mol_to_spec_indices[m_id]
                    exp_mass = mol_consensus_masses[m_id]

                    q_spectra = val_query_embeddings[s_indices].to(device)
                    q_mean = q_spectra.mean(dim=0)

                    if cond["use_neural"] and not cond["use_mass"]:
                        # Neural only: cos_sim (cand_embs @ q_mean)
                        sims = cand_embs_tensor @ q_mean
                        final_scores = sims

                    elif not cond["use_neural"] and cond["use_mass"]:
                        # Mass only: exp(-0.5 * (ppm / sigma)^2)
                        delta_m = torch.abs(cand_masses_tensor - exp_mass)
                        ppm = (delta_m / exp_mass) * 1e6
                        mass_scores = torch.exp(-0.5 * (ppm / cond["sigma"]) ** 2)
                        final_scores = mass_scores

                    elif cond["use_neural"] and cond["use_mass"]:
                        # Neural + Mass: (cos_sim + 1.0) * mass_score
                        sims = cand_embs_tensor @ q_mean
                        delta_m = torch.abs(cand_masses_tensor - exp_mass)
                        ppm = (delta_m / exp_mass) * 1e6
                        mass_scores = torch.exp(-0.5 * (ppm / cond["sigma"]) ** 2)
                        final_scores = (sims + 1.0) * mass_scores

                    top25 = torch.topk(final_scores, k=25).indices.cpu().tolist()
                    rank = top25.index(true_c_idx) + 1 if true_c_idx in top25 else 26
                    ranks.append(rank)

                t_cost = time.time() - t0
                m = compute_retrieval_metrics(ranks, top_k_list=[1, 5, 10, 25])
                rec = {
                    "Checkpoint": ckpt_label,
                    "Condition": cond_name,
                    "MRR@25": m["mrr@25"],
                    "Recall@1": m["recall@1"],
                    "Recall@5": m["recall@5"],
                    "Recall@10": m["recall@10"],
                    "Recall@25": m["recall@25"],
                    "Full_MRR": m["mrr"],
                    "Runtime_s": round(t_cost, 2),
                }
                all_ablation_results.append(rec)
                logger.info(
                    f"[{ckpt_label}] {cond_name:<34} | MRR@25: {m['mrr@25']:.4f} | "
                    f"R@1: {m['recall@1']*100:5.2f}% | R@5: {m['recall@5']*100:5.2f}% | "
                    f"R@10: {m['recall@10']*100:5.2f}% | R@25: {m['recall@25']*100:5.2f}% | "
                    f"Cost: {t_cost:.2f}s"
                )

    df_out = pd.DataFrame(all_ablation_results)
    print("\n" + "=" * 95)
    print("MASS-SCORING ABLATION & CHECKPOINT AUDIT TABLE")
    print("=" * 95)
    print(df_out[["Checkpoint", "Condition", "MRR@25", "Recall@1", "Recall@5", "Recall@10", "Recall@25"]].to_string(index=False))

    out_file = Path("outputs/mass_ablation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_ablation_results, f, indent=2)
    logger.info(f"Saved results to: {out_file}")


if __name__ == "__main__":
    run_ablation()

