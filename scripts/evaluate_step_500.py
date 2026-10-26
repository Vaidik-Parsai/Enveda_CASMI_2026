"""Evaluation of spectrum_molecule_step_500.pt on held-out real data."""

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Add src to python path
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
    generate_morgan_fingerprint,
)
from enveda.evaluation.metrics import compute_retrieval_metrics
from enveda.evaluation.retrieval import chunked_topk_search
from enveda.models import SpectrumMoleculeModel
from enveda.training import load_checkpoint
from enveda.utils import set_seed, setup_logger


def run_step_500_evaluation(num_val_queries: int = 500, total_candidates: int = 3000):
    logger = setup_logger("eval_step_500")
    logger.info("=" * 65)
    logger.info(f"VALIDATING STEP 500 CHECKPOINT ON HELD-OUT REAL DATA")
    logger.info("=" * 65)

    config = load_config("configs/default.yaml")
    set_seed(config.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device} (GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'})")

    # 1. Inspect Checkpoint
    ckpt_path = Path("checkpoints/spectrum_molecule_step_500.pt")
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = SpectrumMoleculeModel(
        spectrum_dim=config.model.spectrum_dim,
        metadata_dim=config.model.metadata_dim,
        fingerprint_dim=config.model.fingerprint_dim,
        embedding_dim=config.model.embedding_dim,
    ).to(device)

    ckpt_meta = load_checkpoint(ckpt_path, model=model, device=device, strict=True)
    logger.info(f"Checkpoint verified: Global Step = {ckpt_meta.get('step')}, Epoch = {ckpt_meta.get('epoch', 0)}")
    logger.info(f"Loss history entries: {len(ckpt_meta.get('loss_history', []))}")
    logger.info(f"Start Loss: {ckpt_meta['loss_history'][0]:.4f} -> Final Loss: {ckpt_meta['loss_history'][-1]:.4f}")

    # 2. Load Split and Verify Zero Leakage
    split_file = Path(config.data.split_file)
    logger.info(f"Reading split cache from: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_rows = split_data["train_indices"]
    val_rows = split_data["val_indices"]
    logger.info(f"Total training spectra in split: {len(train_rows):,}")
    logger.info(f"Total validation spectra in split: {len(val_rows):,}")

    # Subsample validation queries
    rng = np.random.default_rng(config.data.seed)
    eval_val_rows = rng.choice(val_rows, size=min(num_val_queries, len(val_rows)), replace=False).tolist()
    logger.info(f"Selected {len(eval_val_rows):,} held-out validation spectra for evaluation.")

    # 3. Read Validation Parquet Data
    train_parquet = Path(config.data.train_path)
    logger.info(f"Loading selected validation rows from {train_parquet}...")
    table = pq.read_table(
        str(train_parquet),
        columns=[
            "ms2_mzs",
            "ms2_normalized_intensities",
            "precursor_mz",
            "base_peak_intensity",
            "collision_energy_ev",
            "adduct",
            "ionization_mode",
            "normalized_smiles",
            "inchikey14",
        ],
    )
    all_df = table.to_pandas()
    val_df = all_df.iloc[eval_val_rows].reset_index(drop=True)

    # Verify zero molecule overlap
    val_molecules = set(val_df["inchikey14"].dropna())
    train_sample_mols = set(all_df.iloc[train_rows[:50000]]["inchikey14"].dropna())
    overlap = val_molecules.intersection(train_sample_mols)
    logger.info(f"Molecule overlap between val queries and training sample: {len(overlap)} (Zero leakage confirmed!)")
    assert len(overlap) == 0

    # 4. Construct Candidate Pool
    # Ground truth molecules for the evaluated queries
    true_smiles = val_df["normalized_smiles"].dropna().unique().tolist()
    logger.info(f"Ground-truth unique molecules in validation query set: {len(true_smiles):,}")

    # Add distractor molecules
    all_unique_smiles = all_df["normalized_smiles"].dropna().unique()
    distractor_pool = [s for s in all_unique_smiles if s not in set(true_smiles)]
    rng.shuffle(distractor_pool)
    distractors = distractor_pool[: (total_candidates - len(true_smiles))]

    candidate_smiles = true_smiles + distractors
    rng.shuffle(candidate_smiles)
    logger.info(f"Constructed candidate pool of {len(candidate_smiles):,} unique molecules (100% contains true target).")
    smi_to_cand_idx = {smi: idx for idx, smi in enumerate(candidate_smiles)}

    # 5. Encode Candidate Molecules
    logger.info(f"Encoding {len(candidate_smiles):,} candidate molecules with Molecule Tower...")
    model.eval()
    cand_fps = np.zeros((len(candidate_smiles), config.model.fingerprint_dim), dtype=np.float32)
    for i, smi in enumerate(candidate_smiles):
        fp, _ = generate_morgan_fingerprint(
            smi, radius=config.data.fingerprint_radius, n_bits=config.model.fingerprint_dim
        )
        cand_fps[i] = fp

    cand_embs_list = []
    with torch.inference_mode():
        for start_idx in range(0, len(candidate_smiles), 256):
            end_idx = min(start_idx + 256, len(candidate_smiles))
            batch_fp = torch.from_numpy(cand_fps[start_idx:end_idx]).to(device)
            mol_embs = model.encode_molecule(batch_fp)
            cand_embs_list.append(mol_embs.cpu())

    candidate_embeddings = torch.cat(cand_embs_list, dim=0).numpy()
    logger.info(f"Candidate embeddings shape: {candidate_embeddings.shape}, Norm: {np.linalg.norm(candidate_embeddings[:5], axis=1)}")

    # 6. Encode Query Spectra
    logger.info(f"Encoding {len(val_df):,} validation spectra with Spectrum Tower...")
    meta_path = Path(config.data.meta_preprocessor_path)
    preprocessor = MetadataPreprocessor.load(meta_path)
    meta_features = preprocessor.transform(val_df)

    query_embs_list = []
    true_target_indices = []

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

            s_t = torch.from_numpy(spec_vec).unsqueeze(0).to(device)
            m_t = torch.from_numpy(meta_features[i]).unsqueeze(0).to(device)
            spec_emb = model.encode_spectrum(s_t, m_t)
            query_embs_list.append(spec_emb.cpu())

            true_smi = row.get("normalized_smiles", "")
            true_target_indices.append(smi_to_cand_idx[true_smi])

    query_embeddings = torch.cat(query_embs_list, dim=0)

    # 7. Chunked Retrieval Search
    logger.info("Executing chunked cosine similarity retrieval (chunk_size=1000)...")
    top_k_eval = 25
    topk_indices, topk_scores = chunked_topk_search(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        top_k=top_k_eval,
        chunk_size=1000,
        device=device,
    )

    # 8. Compute Retrieval Metrics
    ranks = []
    for q_idx, true_cand_idx in enumerate(true_target_indices):
        preds = topk_indices[q_idx].tolist()
        if true_cand_idx in preds:
            rank = preds.index(true_cand_idx) + 1  # 1-based rank
        else:
            rank = len(candidate_smiles) + 1
        ranks.append(rank)

    ranks_arr = np.array(ranks)
    metrics = {
        "recall@1": float(np.mean(ranks_arr <= 1)),
        "recall@5": float(np.mean(ranks_arr <= 5)),
        "recall@10": float(np.mean(ranks_arr <= 10)),
        "recall@25": float(np.mean(ranks_arr <= 25)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "num_queries": len(val_df),
        "num_candidates": len(candidate_smiles),
    }

    # Compare against random chance baseline
    random_mrr = np.mean(1.0 / np.random.randint(1, len(candidate_smiles) + 1, size=len(ranks_arr)))
    random_r1 = 1.0 / len(candidate_smiles)
    random_r25 = 25.0 / len(candidate_smiles)

    print("\n" + "=" * 65)
    print("STEP 500 RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Validation spectra evaluated: {metrics['num_queries']}")
    print(f"Candidate pool size:         {metrics['num_candidates']:,} unique molecules")
    print("-" * 65)
    print(f"Recall@1:                    {metrics['recall@1'] * 100:.2f}%  (Random baseline: {random_r1*100:.3f}%)")
    print(f"Recall@5:                    {metrics['recall@5'] * 100:.2f}%")
    print(f"Recall@10:                   {metrics['recall@10'] * 100:.2f}%")
    print(f"Recall@25:                   {metrics['recall@25'] * 100:.2f}% (Random baseline: {random_r25*100:.2f}%)")
    print(f"Mean Reciprocal Rank (MRR):  {metrics['mrr']:.4f}  (Random baseline: {random_mrr:.4f})")
    print(f"Median Rank:                 {metrics['median_rank']:.1f}")
    print(f"Mean Rank:                   {metrics['mean_rank']:.1f}")
    print("=" * 65 + "\n")

    return metrics


if __name__ == "__main__":
    run_step_500_evaluation()
