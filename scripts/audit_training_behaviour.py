"""Comprehensive audit script for Phase 6A: Training Behaviour Audit.

Analyzes checkpoints 5k, 10k, 15k, 20k, batch composition, optimizer state,
embedding distributions, dimensional collapse, and positive/negative cosine margin.
"""

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
    StreamingParquetDataset,
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
from enveda.training.checkpointing import load_checkpoint
from enveda.training.losses import compute_batch_tanimoto, multi_positive_contrastive_loss
from enveda.utils import setup_logger


def compute_effective_rank(embeddings: torch.Tensor) -> float:
    """Compute the effective rank (Roy & Vetterli, 2007) of embedding matrix.
    
    Measures dimensional collapse: if all dimensions are equally utilized,
    effective rank equals embedding_dim (128). If collapsed, rank << 128.
    """
    # Center embeddings
    X = embeddings - embeddings.mean(dim=0, keepdim=True)
    # SVD
    _, S, _ = torch.linalg.svd(X, full_matrices=False)
    # Singular value probabilities
    p = S / (S.sum() + 1e-12)
    # Shannon entropy
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    # Effective rank = exp(entropy)
    return float(torch.exp(entropy).item())


def run_training_audit():
    logger = setup_logger("training_audit")
    logger.info("=" * 80)
    logger.info("PHASE 6A: SYSTEMATIC TRAINING BEHAVIOUR AUDIT")
    logger.info("=" * 80)

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # =========================================================================
    # TASK 1 & 2: BATCH COMPOSITION & HARD NEGATIVE AUDIT
    # =========================================================================
    logger.info("-" * 80)
    logger.info("TASK 2: BATCH COMPOSITION & HARD-NEGATIVE MINING AUDIT")
    logger.info("-" * 80)

    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    train_rows, val_rows = create_or_load_group_split(train_parquet, split_path)

    preprocessor = MetadataPreprocessor.load(config.data.meta_preprocessor_path)
    train_row_set = set(train_rows)
    dataset = StreamingParquetDataset(
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

    num_sample_batches = 50
    logger.info(f"Sampling {num_sample_batches} training batches from StreamingParquetDataset...")

    batch_unique_mols = []
    batch_pos_pairs = []
    batch_neg_tanimoto_all = []
    batch_hard_neg_counts = []  # T >= 0.4
    batch_very_hard_counts = []  # T >= 0.6
    batch_mass_compatible_hard_neg_pct = []  # T >= 0.4 and within 20 ppm

    batch_iter = iter(dataset)
    with torch.no_grad():
        for b_idx in range(num_sample_batches):
            batch = next(batch_iter)
            mol_ids = batch["molecule_ids"]  # (batch_size,)
            fps = batch["fingerprint"]  # (batch_size, 2048)
            pmz = batch["metadata"][:, 0]  # rough precursor mz column

            bsz = len(mol_ids)
            unique_ids = torch.unique(mol_ids)
            batch_unique_mols.append(len(unique_ids))

            pos_mask = mol_ids[:, None] == mol_ids[None, :]
            num_pos_pairs = int((pos_mask.sum().item() - bsz) / 2)  # off-diagonal positives
            batch_pos_pairs.append(num_pos_pairs)

            tanimoto = compute_batch_tanimoto(fps)
            neg_mask = ~pos_mask
            neg_tanimoto = tanimoto[neg_mask].cpu().numpy()
            batch_neg_tanimoto_all.extend(neg_tanimoto[:1000])

            # Hard negatives per anchor (T >= 0.4)
            hard_neg_mask = neg_mask & (tanimoto >= 0.4)
            very_hard_neg_mask = neg_mask & (tanimoto >= 0.6)
            hard_per_anchor = hard_neg_mask.sum(dim=1).float().mean().item()
            very_hard_per_anchor = very_hard_neg_mask.sum(dim=1).float().mean().item()
            batch_hard_neg_counts.append(hard_per_anchor)
            batch_very_hard_counts.append(very_hard_per_anchor)

    logger.info("Batch Composition Results (across 50 batches):")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Mean unique molecules per batch: {np.mean(batch_unique_mols):.1f} / {config.training.batch_size} ({np.mean(batch_unique_mols)/config.training.batch_size*100:.1f}%)")
    logger.info(f"  Mean off-diagonal positive pairs per batch: {np.mean(batch_pos_pairs):.2f}")
    logger.info(f"  Mean hard negatives (T >= 0.4) per anchor: {np.mean(batch_hard_neg_counts):.2f}")
    logger.info(f"  Mean very hard negatives (T >= 0.6) per anchor: {np.mean(batch_very_hard_counts):.2f}")
    logger.info(f"  Negative Tanimoto distribution: Mean = {np.mean(batch_neg_tanimoto_all):.4f}, Median = {np.median(batch_neg_tanimoto_all):.4f}, Max = {np.max(batch_neg_tanimoto_all):.4f}")
    logger.info(f"  Percent of negative pairs with T >= 0.3: {np.mean(np.array(batch_neg_tanimoto_all) >= 0.3)*100:.2f}%")
    logger.info(f"  Percent of negative pairs with T >= 0.4: {np.mean(np.array(batch_neg_tanimoto_all) >= 0.4)*100:.2f}%")
    logger.info(f"  Percent of negative pairs with T >= 0.5: {np.mean(np.array(batch_neg_tanimoto_all) >= 0.5)*100:.2f}%")

    # =========================================================================
    # TASK 3: OPTIMIZER & SCALER AUDIT ACROSS CHECKPOINTS
    # =========================================================================
    logger.info("-" * 80)
    logger.info("TASK 3: OPTIMIZER, LEARNING RATE, & GRADSCALER AUDIT")
    logger.info("-" * 80)

    checkpoint_paths = [
        ("Step 5k", Path("checkpoints/spectrum_molecule_step_5000.pt")),
        ("Step 10k", Path("checkpoints/spectrum_molecule_step_10000.pt")),
        ("Step 15k", Path("checkpoints/spectrum_molecule_step_15000.pt")),
        ("Step 20k", Path("checkpoints/spectrum_molecule_step_20000.pt")),
    ]

    opt_audit_results = {}
    for label, ckpt_p in checkpoint_paths:
        ckpt_data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        step = ckpt_data.get("step")
        loss_hist = ckpt_data.get("loss_history", [])
        opt_state = ckpt_data.get("optimizer")
        scaler_state = ckpt_data.get("scaler")

        lr = None
        weight_decay = None
        has_momentum = False
        if opt_state and "param_groups" in opt_state:
            lr = opt_state["param_groups"][0].get("lr")
            weight_decay = opt_state["param_groups"][0].get("weight_decay")
            has_momentum = len(opt_state.get("state", {})) > 0

        scale = scaler_state.get("scale") if scaler_state else None
        recent_loss = float(np.mean(loss_hist[-50:])) if len(loss_hist) >= 50 else None

        opt_audit_results[label] = {
            "saved_step": step,
            "loss_records": len(loss_hist),
            "recent_mean_loss": recent_loss,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "has_momentum_buffers": has_momentum,
            "grad_scaler_scale": scale,
        }
        logger.info(f"[{label}] Step: {step} | LR: {lr} | Weight Decay: {weight_decay} | Scaler Scale: {scale} | Recent Train Loss: {recent_loss:.4f} | Loss History Length: {len(loss_hist)}")

    # =========================================================================
    # TASK 4: CHECKPOINT COMPARISON (REPRESENTATION SPACE & MARGINS)
    # =========================================================================
    logger.info("-" * 80)
    logger.info("TASK 4: REPRESENTATION SPACE & POS/NEG MARGIN AUDIT ACROSS CHECKPOINTS")
    logger.info("-" * 80)

    # 1. Load Validation Data (400 molecules, 17,267 spectra)
    val_table = pq.read_table(
        str(train_parquet),
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

    # Pre-parse validation spectra
    meta_features = preprocessor.transform(val_df)
    spec_vectors = []
    fps_list = []
    mol_id_ints = []
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
        smi = row["normalized_smiles"]
        from enveda.data.preprocessing import generate_morgan_fingerprint
        fp, _ = generate_morgan_fingerprint(smi, radius=config.data.fingerprint_radius, n_bits=config.model.fingerprint_dim)
        fps_list.append(fp)
        mol_id_ints.append(hash(row["inchikey14"]) & 0x7FFFFFFFFFFFFFFF)

    spec_vectors = np.array(spec_vectors, dtype=np.float32)
    fps_array = np.array(fps_list, dtype=np.float32)
    mol_id_tensor = torch.tensor(mol_id_ints, dtype=torch.int64)

    # Precomputed candidate exact masses
    candidate_masses = np.load("outputs/candidate_bank/candidate_exact_masses.npy")
    cand_masses_tensor = torch.from_numpy(candidate_masses).to(device)

    # Manifest SMILES
    with open("outputs/candidate_bank/candidate_manifest.json", "r", encoding="utf-8") as f:
        manifest_smiles = json.load(f)["smiles"]
    smiles_to_c_idx = {smi: idx for idx, smi in enumerate(manifest_smiles)}

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

    comparison_results = []

    for label, ckpt_p in checkpoint_paths:
        logger.info(f"\nAnalyzing representations for {label} ({ckpt_p.name})...")
        model = SpectrumMoleculeModel(
            spectrum_dim=config.model.spectrum_dim,
            metadata_dim=config.model.metadata_dim,
            fingerprint_dim=config.model.fingerprint_dim,
            embedding_dim=config.model.embedding_dim,
        ).to(device)
        load_checkpoint(ckpt_p, model=model, device=device)
        model.eval()

        # 1. Compute Validation Loss on 1000 validation pairs
        val_losses = []
        with torch.no_grad():
            eval_batch_size = 128
            for v_i in range(0, min(1280, len(val_df)), eval_batch_size):
                s_b = torch.from_numpy(spec_vectors[v_i : v_i + eval_batch_size]).to(device)
                m_b = torch.from_numpy(meta_features[v_i : v_i + eval_batch_size]).to(device)
                fp_b = torch.from_numpy(fps_array[v_i : v_i + eval_batch_size]).to(device)
                ids_b = mol_id_tensor[v_i : v_i + eval_batch_size].to(device)

                s_emb, mol_emb = model(s_b, m_b, fp_b)
                v_loss = multi_positive_contrastive_loss(s_emb, mol_emb, ids_b, temperature=0.07)
                val_losses.append(v_loss.item())
        mean_val_loss = float(np.mean(val_losses))

        # 2. Encode all validation query spectra
        q_embs_list = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                s_b = torch.from_numpy(spec_vectors[i : i + batch_size]).to(device)
                m_b = torch.from_numpy(meta_features[i : i + batch_size]).to(device)
                q_embs_list.append(model.encode_spectrum(s_b, m_b))
        val_spec_embeddings = torch.cat(q_embs_list, dim=0)  # (17267, 128)

        # 3. Encode the 400 unique validation molecules
        unique_fps = []
        for m_id in unique_inchikeys:
            sub = val_df[val_df["inchikey14"] == m_id]
            smi = sub["normalized_smiles"].iloc[0]
            from enveda.data.preprocessing import generate_morgan_fingerprint
            fp, _ = generate_morgan_fingerprint(smi, radius=config.data.fingerprint_radius, n_bits=config.model.fingerprint_dim)
            unique_fps.append(fp)
        unique_fps_tensor = torch.from_numpy(np.array(unique_fps, dtype=np.float32)).to(device)

        with torch.no_grad():
            unique_mol_embeddings = model.encode_molecule(unique_fps_tensor)  # (400, 128)

        # 4. Measure Embedding Distributions & Effective Ranks
        spec_eff_rank = compute_effective_rank(val_spec_embeddings[:2000])
        mol_eff_rank = compute_effective_rank(unique_mol_embeddings)

        spec_norms = torch.norm(val_spec_embeddings, p=2, dim=1).mean().item()
        mol_norms = torch.norm(unique_mol_embeddings, p=2, dim=1).mean().item()

        # 5. Measure Cosine Similarities: Positive vs. Negative Pairs
        # Positive: spectrum embedding vs its true molecule embedding
        pos_sims = []
        neg_sims = []
        with torch.no_grad():
            for m_idx, m_id in enumerate(unique_inchikeys):
                s_indices = mol_to_spec_indices[m_id]
                m_emb = unique_mol_embeddings[m_idx]  # (128,)
                s_embs = val_spec_embeddings[s_indices]  # (N, 128)
                pos_cos = (s_embs @ m_emb).cpu().numpy()
                pos_sims.extend(pos_cos)

                # Negatives: against other 399 molecules
                neg_cos = (s_embs @ unique_mol_embeddings.T).cpu().numpy()  # (N, 400)
                # Mask out positive column
                neg_cos_filtered = np.delete(neg_cos, m_idx, axis=1).flatten()
                neg_sims.extend(neg_cos_filtered[:200])

        pos_sims = np.array(pos_sims)
        neg_sims = np.array(neg_sims)
        mean_pos = float(np.mean(pos_sims))
        mean_neg = float(np.mean(neg_sims))
        margin = mean_pos - mean_neg

        # 6. Candidate Bank Embeddings & Retrieval Performance
        # Load or generate candidate bank embeddings for this exact checkpoint
        ckpt_stem = ckpt_p.stem
        cand_dir = Path(f"outputs/candidate_bank_{ckpt_stem}")
        emb_file = cand_dir / "candidate_embeddings.npy"
        if not emb_file.is_file():
            if ckpt_stem == "spectrum_molecule_step_5000" and Path("outputs/candidate_bank/candidate_embeddings.npy").is_file():
                emb_file = Path("outputs/candidate_bank/candidate_embeddings.npy")
            else:
                from enveda.evaluation.retrieval import build_candidate_bank
                cand_dir.mkdir(parents=True, exist_ok=True)
                build_candidate_bank(
                    smiles_list=manifest_smiles,
                    model=model,
                    output_dir=cand_dir,
                    batch_size=512,
                    device=device,
                )
        cand_embs_np = np.load(str(emb_file), mmap_mode="r")
        cand_embs_tensor = torch.from_numpy(cand_embs_np[:].copy()).to(device)

        # Retrieval Benchmark on 400 molecules:
        # A) Pure Neural Mean
        # B) Mass-Aware Mean (sigma = 20 ppm)
        # C) Mass-Aware Mean (sigma = 30 ppm)
        ranks_un = []
        ranks_m20 = []
        ranks_m30 = []

        with torch.no_grad():
            for m_id in unique_inchikeys:
                true_c_idx = mol_true_cand_idx[m_id]
                s_indices = mol_to_spec_indices[m_id]
                exp_mass = mol_consensus_masses[m_id]

                q_spectra = val_spec_embeddings[s_indices].to(device)
                q_mean = q_spectra.mean(dim=0)
                raw_sims = cand_embs_tensor @ q_mean

                # Unconstrained
                top25_un = torch.topk(raw_sims, k=25).indices.cpu().tolist()
                ranks_un.append(top25_un.index(true_c_idx) + 1 if true_c_idx in top25_un else 26)

                # Mass delta & ppm
                delta_m = torch.abs(cand_masses_tensor - exp_mass)
                ppm = (delta_m / exp_mass) * 1e6

                # Mass 20 ppm
                m_score20 = torch.exp(-0.5 * (ppm / 20.0) ** 2)
                top25_m20 = torch.topk((raw_sims + 1.0) * m_score20, k=25).indices.cpu().tolist()
                ranks_m20.append(top25_m20.index(true_c_idx) + 1 if true_c_idx in top25_m20 else 26)

                # Mass 30 ppm
                m_score30 = torch.exp(-0.5 * (ppm / 30.0) ** 2)
                top25_m30 = torch.topk((raw_sims + 1.0) * m_score30, k=25).indices.cpu().tolist()
                ranks_m30.append(top25_m30.index(true_c_idx) + 1 if true_c_idx in top25_m30 else 26)

        m_un_res = compute_retrieval_metrics(ranks_un, top_k_list=[1, 5, 10, 25])
        m_20_res = compute_retrieval_metrics(ranks_m20, top_k_list=[1, 5, 10, 25])
        m_30_res = compute_retrieval_metrics(ranks_m30, top_k_list=[1, 5, 10, 25])

        row_res = {
            "Checkpoint": label,
            "Step": ckpt_p.stem.split("_")[-1],
            "Recent_Train_Loss": opt_audit_results[label]["recent_mean_loss"],
            "Validation_Loss": mean_val_loss,
            "Spec_Effective_Rank": spec_eff_rank,
            "Mol_Effective_Rank": mol_eff_rank,
            "Pos_Cosine_Mean": mean_pos,
            "Neg_Cosine_Mean": mean_neg,
            "Cosine_Margin": margin,
            "Unconstrained_MRR@25": m_un_res["mrr@25"],
            "Unconstrained_Recall@25": m_un_res["recall@25"],
            "Mass20_MRR@25": m_20_res["mrr@25"],
            "Mass20_Recall@1": m_20_res["recall@1"],
            "Mass20_Recall@5": m_20_res["recall@5"],
            "Mass20_Recall@10": m_20_res["recall@10"],
            "Mass20_Recall@25": m_20_res["recall@25"],
            "Mass30_MRR@25": m_30_res["mrr@25"],
            "Mass30_Recall@25": m_30_res["recall@25"],
        }
        comparison_results.append(row_res)

        logger.info(
            f"  {label:<9} | TrainLoss: {row_res['Recent_Train_Loss']:.4f} | ValLoss: {mean_val_loss:.4f} | "
            f"SpecRank: {spec_eff_rank:5.1f} | MolRank: {mol_eff_rank:5.1f} | "
            f"PosCos: {mean_pos:+.3f} | NegCos: {mean_neg:+.3f} | Margin: {margin:+.3f} | "
            f"Mass20 MRR@25: {m_20_res['mrr@25']:.4f} | Mass20 R@25: {m_20_res['recall@25']*100:5.2f}%"
        )

    df_comp = pd.DataFrame(comparison_results)
    print("\n" + "=" * 100)
    print("PHASE 6A: CHECKPOINT REPRESENTATION & RETRIEVAL AUDIT")
    print("=" * 100)
    print(df_comp[["Checkpoint", "Recent_Train_Loss", "Validation_Loss", "Spec_Effective_Rank", "Mol_Effective_Rank", "Pos_Cosine_Mean", "Neg_Cosine_Mean", "Cosine_Margin", "Mass20_MRR@25", "Mass20_Recall@25"]].to_string(index=False))

    audit_full = {
        "batch_composition": {
            "mean_unique_mols": float(np.mean(batch_unique_mols)),
            "mean_off_diagonal_pos_pairs": float(np.mean(batch_pos_pairs)),
            "mean_hard_neg_per_anchor": float(np.mean(batch_hard_neg_counts)),
            "mean_very_hard_neg_per_anchor": float(np.mean(batch_very_hard_counts)),
            "pct_neg_tanimoto_ge_03": float(np.mean(np.array(batch_neg_tanimoto_all) >= 0.3) * 100),
            "pct_neg_tanimoto_ge_04": float(np.mean(np.array(batch_neg_tanimoto_all) >= 0.4) * 100),
            "pct_neg_tanimoto_ge_05": float(np.mean(np.array(batch_neg_tanimoto_all) >= 0.5) * 100),
            "neg_tanimoto_mean": float(np.mean(batch_neg_tanimoto_all)),
            "neg_tanimoto_median": float(np.median(batch_neg_tanimoto_all)),
        },
        "optimizer_audit": opt_audit_results,
        "checkpoint_comparison": comparison_results,
    }

    out_file = Path("outputs/phase_6a_training_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_full, f, indent=2)
    logger.info(f"\nSaved Phase 6A training audit results to {out_file}")


if __name__ == "__main__":
    run_training_audit()

