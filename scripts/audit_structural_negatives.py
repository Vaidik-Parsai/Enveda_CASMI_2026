"""Audit structural hard-negative distribution and empirical Tanimoto thresholds."""

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
from enveda.data import create_or_load_group_split
from enveda.data.preprocessing import generate_morgan_fingerprint


def main():
    config = load_config("configs/default.yaml")
    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    train_rows, val_rows = create_or_load_group_split(train_parquet, split_path)

    print(f"Loaded split: {len(train_rows):,} train spectra | {len(val_rows):,} val spectra")

    # Load 50,000 training rows to get unique molecules
    table = pq.read_table(
        str(train_parquet),
        columns=["normalized_smiles", "inchikey14"],
    )
    df_sample = table.to_pandas().iloc[train_rows[:50000]].reset_index(drop=True)
    unique_mols = df_sample.drop_duplicates(subset=["inchikey14"]).reset_index(drop=True)
    print(f"Extracted {len(unique_mols):,} unique training molecules for candidate pool.")

    # Compute fingerprints
    fps_list = []
    smiles_list = unique_mols["normalized_smiles"].tolist()
    inchikeys = unique_mols["inchikey14"].tolist()
    valid_indices = []

    for idx, s in enumerate(smiles_list):
        fp, ok = generate_morgan_fingerprint(s, radius=2, n_bits=2048)
        if ok:
            fps_list.append(fp)
            valid_indices.append(idx)

    fps_arr = np.array(fps_list, dtype=np.uint8)
    pool_inchikeys = [inchikeys[i] for i in valid_indices]
    num_pool = len(fps_arr)
    print(f"Valid indexed fingerprints in pool: {num_pool:,}")

    # Convert to float tensor on CPU / GPU for fast batch similarity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_tensor = torch.from_numpy((fps_arr > 0.5).astype(np.float32)).to(device)
    fp_sums = fp_tensor.sum(dim=1)

    # Select 500 anchors uniformly
    rng = np.random.default_rng(42)
    anchor_indices = rng.choice(num_pool, size=500, replace=False)

    thresholds = [0.2, 0.3, 0.4, 0.5]
    counts_per_threshold = {t: [] for t in thresholds}
    max_tanimotos = []
    top5_tanimotos = []

    for a_idx in anchor_indices:
        a_fp = fp_tensor[a_idx]
        inter = torch.mv(fp_tensor, a_fp)
        union = fp_sums + a_fp.sum() - inter
        sims = (inter / torch.clamp(union, min=1.0)).cpu().numpy()
        
        # Mask out self / same InChIKey
        a_ikey = pool_inchikeys[a_idx]
        for idx in range(num_pool):
            if pool_inchikeys[idx] == a_ikey:
                sims[idx] = -1.0

        valid_sims = sims[sims >= 0.0]
        sorted_sims = np.sort(valid_sims)[::-1]

        max_tanimotos.append(sorted_sims[0] if len(sorted_sims) > 0 else 0.0)
        top5_tanimotos.append(sorted_sims[:5])

        for t in thresholds:
            n_match = int(np.sum(valid_sims >= t))
            counts_per_threshold[t].append(n_match)

    max_t_arr = np.array(max_tanimotos)
    print("\n" + "=" * 80)
    print("EMPIRICAL MORGAN TANIMOTO SIMILARITY DISTRIBUTION (500 ANCHORS)")
    print("=" * 80)
    print(f"Max Tanimoto to nearest neighbor in pool:")
    print(f"  Min:    {np.min(max_t_arr):.4f}")
    print(f"  25th%:  {np.percentile(max_t_arr, 25):.4f}")
    print(f"  Median: {np.median(max_t_arr):.4f}")
    print(f"  Mean:   {np.mean(max_t_arr):.4f} +/- {np.std(max_t_arr):.4f}")
    print(f"  75th%:  {np.percentile(max_t_arr, 75):.4f}")
    print(f"  Max:    {np.max(max_t_arr):.4f}")

    print("\nCandidate Availability per Anchor by Threshold:")
    print(f"{'Threshold':<10} | {'Mean Matches':<14} | {'Median Matches':<16} | {'Anchors with >= 1 Match':<25}")
    print("-" * 75)
    for t in thresholds:
        arr = np.array(counts_per_threshold[t])
        pct_with_match = 100.0 * np.mean(arr > 0)
        print(f"T >= {t:<6} | {np.mean(arr):<14.1f} | {np.median(arr):<16.1f} | {pct_with_match:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

