"""Evaluation of Phase 6B Run C checkpoints against the current champion and Run A."""

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List
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
from enveda.data.preprocessing import generate_morgan_fingerprint
from enveda.evaluation.mass_filter import (
    calculate_exact_monoisotopic_mass,
    get_molecule_consensus_neutral_mass,
)
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.retrieval import build_candidate_bank
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.training.losses import multi_positive_contrastive_loss
from enveda.utils import setup_logger


def compute_effective_rank(embeddings: torch.Tensor) -> float:
    """Compute effective rank exp(entropy of singular value distribution)."""
    with torch.no_grad():
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        _, s, _ = torch.svd(centered)
        s = s[s > 1e-7]
        p = s / s.sum()
        entropy = -torch.sum(p * torch.log(p)).item()
        return float(np.exp(entropy))


def evaluate_run_c():
    logger = setup_logger("eval_phase6b_run_c")
    logger.info("=" * 80)
    logger.info("EVALUATION OF PHASE 6B RUN C CHECKPOINTS (STEPS 1000 - 5000)")
    logger.info("=" * 80)

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    # 1. Candidate Bank Manifest & Masses
    base_manifest_path = Path("outputs/candidate_bank/candidate_manifest.json")
    with open(base_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    candidate_smiles = manifest["smiles"]
    num_candidates = len(candidate_smiles)
    logger.info(f"Candidate bank universe: {num_candidates:,} SMILES.")

    candidate_masses = np.load("outputs/candidate_bank/candidate_exact_masses.npy")
    cand_masses_tensor = torch.from_numpy(candidate_masses).to(device)
    smiles_to_c_idx = {smi: idx for idx, smi in enumerate(candidate_smiles)}

    # 2. Validation Data (400 molecules)
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
    mol_survives_mass_screen = {}

    for m_id in unique_inchikeys:
        sub_df = val_df[val_df["inchikey14"] == m_id]
        true_smi = sub_df["normalized_smiles"].iloc[0]
        true_c_idx = smiles_to_c_idx[true_smi]
        mol_true_cand_idx[m_id] = true_c_idx
        pmzs = sub_df["precursor_mz"].tolist()
        adds = sub_df["adduct"].tolist()
        modes = sub_df["ionization_mode"].tolist()
        c_mass = get_molecule_consensus_neutral_mass(pmzs, adds, modes)
        if c_mass is None:
            c_mass = calculate_exact_monoisotopic_mass(true_smi)
        mol_consensus_masses[m_id] = c_mass

        # Check target survival under mass screening (+/- 50 ppm window)
        true_exact_mass = candidate_masses[true_c_idx]
        delta_ppm = abs(true_exact_mass - c_mass) / c_mass * 1e6
        mol_survives_mass_screen[m_id] = (delta_ppm <= 50.0)

    overall_target_survival = sum(mol_survives_mass_screen.values()) / len(unique_inchikeys)
    logger.info(f"Precursor Mass Target Survival Rate (+/- 50 ppm window): {overall_target_survival*100:.2f}%")

    # Pre-parse validation spectra
    logger.info("Pre-parsing validation spectrum vectors and metadata features...")
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
    checkpoints_to_eval = [
        ("Current Champion", "checkpoints/spectrum_molecule_step_5000.pt", 5000, "champion_step_5000"),
        ("Run A Step 5000", "checkpoints/phase6b_run_a/step_5000.pt", 5000, "phase6b_run_a_step_5000"),
        ("Run C Step 1000", "checkpoints/phase6b_run_c/step_1000.pt", 1000, "phase6b_run_c_step_1000"),
        ("Run C Step 2000", "checkpoints/phase6b_run_c/step_2000.pt", 2000, "phase6b_run_c_step_2000"),
        ("Run C Step 3000", "checkpoints/phase6b_run_c/step_3000.pt", 3000, "phase6b_run_c_step_3000"),
        ("Run C Step 4000", "checkpoints/phase6b_run_c/step_4000.pt", 4000, "phase6b_run_c_step_4000"),
        ("Run C Step 5000", "checkpoints/phase6b_run_c/step_5000.pt", 5000, "phase6b_run_c_step_5000"),
    ]

    results = []

    for label, ckpt_path_str, step_num, cand_tag in checkpoints_to_eval:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.is_file():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info("\n" + "-" * 70)
        logger.info(f"Evaluating {label} ({ckpt_path})...")
        logger.info("-" * 70)

        # Initialize model
        model = SpectrumMoleculeModel(
            spectrum_dim=config.model.spectrum_dim,
            metadata_dim=config.model.metadata_dim,
            fingerprint_dim=config.model.fingerprint_dim,
            embedding_dim=config.model.embedding_dim,
        ).to(device)
        load_checkpoint(ckpt_path, model=model, device=device)
        model.eval()

        # Build / load candidate bank embeddings for this checkpoint
        cand_dir = Path(f"outputs/candidate_bank_{cand_tag}")
        cand_emb_file = cand_dir / "candidate_embeddings.npy"

        if not cand_emb_file.is_file():
            logger.info(f"Encoding candidate bank for {label} on GPU...")
            cand_dir.mkdir(parents=True, exist_ok=True)
            t_cand = time.time()
            build_candidate_bank(
                smiles_list=candidate_smiles,
                model=model,
                output_dir=cand_dir,
                batch_size=512,
                device=device,
                fp_radius=config.data.fingerprint_radius,
                fp_n_bits=config.model.fingerprint_dim,
            )
            logger.info(f"Encoded candidate bank in {time.time() - t_cand:.1f}s.")

        cand_embs_np = np.load(str(cand_emb_file), mmap_mode="r")
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

        # Compute effective rank of spectrum embeddings
        eff_rank = compute_effective_rank(val_query_embeddings)

        # Compute validation loss on sample batch of val queries and true molecules
        with torch.no_grad():
            sample_val_spec = torch.from_numpy(spec_vectors[:128]).to(device)
            sample_val_meta = torch.from_numpy(meta_features[:128]).to(device)
            sample_val_spec_emb = model.encode_spectrum(sample_val_spec, sample_val_meta)
            sample_val_fps = []
            for idx in range(128):
                true_smi = val_df["normalized_smiles"].iloc[idx]
                fp_v, _ = generate_morgan_fingerprint(true_smi, 2, 2048)
                sample_val_fps.append(fp_v)
            sample_val_fps_t = torch.from_numpy(np.array(sample_val_fps)).float().to(device)
            sample_val_mol_emb = model.encode_molecule(sample_val_fps_t)
            sample_mol_ids = torch.arange(128, device=device)
            val_loss = multi_positive_contrastive_loss(sample_val_spec_emb, sample_val_mol_emb, sample_mol_ids).item()

        # Evaluate Retrieval: Mass-Aware Mean Scoring (sigma = 20 ppm)
        ranks = []
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        top25_correct = 0
        reciprocal_ranks = []

        with torch.inference_mode():
            for m_id in unique_inchikeys:
                true_c_idx = mol_true_cand_idx[m_id]
                s_indices = mol_to_spec_indices[m_id]
                exp_mass = mol_consensus_masses[m_id]

                q_spectra = val_query_embeddings[s_indices].to(device)
                q_mean = q_spectra.mean(dim=0)
                sims = cand_embs_tensor @ q_mean

                delta_m = torch.abs(cand_masses_tensor - exp_mass)
                ppm = (delta_m / exp_mass) * 1e6
                m_scores = torch.exp(-0.5 * (ppm / 20.0) ** 2)
                comb_sims = (sims + 1.0) * m_scores

                top25 = torch.topk(comb_sims, k=25).indices.cpu().tolist()

                if true_c_idx in top25:
                    rank = top25.index(true_c_idx) + 1
                    ranks.append(rank)
                    reciprocal_ranks.append(1.0 / rank)
                    if rank <= 1:
                        top1_correct += 1
                    if rank <= 5:
                        top5_correct += 1
                    if rank <= 10:
                        top10_correct += 1
                    if rank <= 25:
                        top25_correct += 1
                else:
                    ranks.append(26)
                    reciprocal_ranks.append(0.0)

        num_mols = len(unique_inchikeys)
        mrr25 = float(np.mean(reciprocal_ranks))
        r1 = top1_correct / num_mols
        r5 = top5_correct / num_mols
        r10 = top10_correct / num_mols
        r25 = top25_correct / num_mols
        mean_rank = float(np.mean(ranks))
        median_rank = float(np.median(ranks))

        rec = {
            "Label": label,
            "Step": step_num,
            "Checkpoint": str(ckpt_path),
            "MRR@25": mrr25,
            "Recall@1": r1,
            "Recall@5": r5,
            "Recall@10": r10,
            "Recall@25": r25,
            "Mean_Rank": mean_rank,
            "Median_Rank": median_rank,
            "Target_Survival_Rate": overall_target_survival,
            "Validation_Loss": val_loss,
            "Effective_Rank": eff_rank,
        }
        results.append(rec)

        logger.info(
            f"{label:16s} | MRR@25: {mrr25:.4f} | R@1: {r1*100:5.2f}% | R@5: {r5*100:5.2f}% | "
            f"R@10: {r10*100:5.2f}% | R@25: {r25*100:5.2f}% | TargetSurvival: {overall_target_survival*100:.2f}% | "
            f"EffRank: {eff_rank:.1f} | ValLoss: {val_loss:.4f}"
        )

    # Save results to JSON
    out_file = Path("outputs/phase6b_run_c_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved evaluation results to: {out_file}")

    # Print summary table
    print("\n" + "=" * 92)
    print("PHASE 6B RUN C EVALUATION BENCHMARK vs CURRENT CHAMPION & RUN A")
    print("=" * 92)
    print(f"{'Checkpoint':<20} | {'MRR@25':<8} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<11} | {'Recall@25':<10} | {'TargetSurv':<11} | {'EffRank':<8}")
    print("-" * 105)
    for r in results:
        print(
            f"{r['Label']:<20} | {r['MRR@25']:<8.4f} | {r['Recall@1']*100:<9.2f}% | "
            f"{r['Recall@5']*100:<9.2f}% | {r['Recall@10']*100:<10.2f}% | {r['Recall@25']*100:<9.2f}% | "
            f"{r['Target_Survival_Rate']*100:<10.2f}% | {r['Effective_Rank']:<8.1f}"
        )
    print("=" * 105)


if __name__ == "__main__":
    evaluate_run_c()

