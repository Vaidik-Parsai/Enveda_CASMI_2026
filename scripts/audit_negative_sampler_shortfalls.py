"""Audit negative sampler counters, shortfalls, and distributions over >= 1,000 training anchors."""

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
    ChemistryAwareNegativePool,
    create_or_load_group_split,
)
from enveda.data.preprocessing import generate_morgan_fingerprint
from enveda.evaluation.mass_filter import estimate_neutral_mass


def run_sampler_audit(pool_sizes=[4348, 20000, 50000], num_anchors=1000):
    config = load_config("configs/default.yaml")
    train_parquet = Path(config.data.train_path)
    split_path = Path(config.data.split_file)
    train_rows, val_rows = create_or_load_group_split(train_parquet, split_path)

    print("=" * 80)
    print(f"AUDITING NEGATIVE SAMPLER OVER {num_anchors:,} REPRESENTATIVE TRAINING ANCHORS")
    print("=" * 80)

    # Read training table
    table = pq.read_table(
        str(train_parquet),
        columns=["normalized_smiles", "inchikey14", "precursor_mz", "adduct", "ionization_mode"],
    )
    df_all = table.to_pandas().iloc[train_rows].reset_index(drop=True)

    # Select 1,000 representative anchors from training set
    rng = np.random.default_rng(42)
    anchor_indices = rng.choice(len(df_all), size=num_anchors, replace=False)
    df_anchors = df_all.iloc[anchor_indices].reset_index(drop=True)

    anchor_smiles = df_anchors["normalized_smiles"].tolist()
    anchor_inchikeys = df_anchors["inchikey14"].tolist()
    anchor_mzs = df_anchors["precursor_mz"].astype(float).tolist()
    anchor_adducts = df_anchors["adduct"].tolist()
    anchor_modes = df_anchors["ionization_mode"].tolist()

    # Pre-generate fingerprints for anchors
    anchor_fps = []
    valid_anchor_mask = []
    for s in anchor_smiles:
        fp, ok = generate_morgan_fingerprint(s, radius=2, n_bits=2048)
        anchor_fps.append(fp if ok else np.zeros(2048, dtype=np.uint8))
        valid_anchor_mask.append(ok)
    anchor_fps = np.array(anchor_fps, dtype=np.uint8)

    print(f"Loaded {num_anchors:,} anchors. Valid fingerprints: {sum(valid_anchor_mask):,}")

    for p_size in pool_sizes:
        print(f"\n{'='*80}")
        print(f"AUDITING CANDIDATE POOL SIZE: {p_size:,} UNIQUE MOLECULES")
        print(f"{'='*80}")

        # Extract unique molecules for pool
        unique_mols = df_all.drop_duplicates(subset=["inchikey14"]).head(p_size).reset_index(drop=True)
        pool_smiles = unique_mols["normalized_smiles"].tolist()
        pool_inchikeys = unique_mols["inchikey14"].tolist()

        fps_list = []
        valid_pool_smiles = []
        valid_pool_inchikeys = []
        for s, ikey in zip(pool_smiles, pool_inchikeys):
            fp, ok = generate_morgan_fingerprint(s, radius=2, n_bits=2048)
            if ok:
                fps_list.append(fp)
                valid_pool_smiles.append(s)
                valid_pool_inchikeys.append(ikey)

        pool_fps = np.array(fps_list, dtype=np.uint8)
        pool = ChemistryAwareNegativePool(
            training_smiles=valid_pool_smiles,
            training_inchikeys=valid_pool_inchikeys,
            training_fingerprints=pool_fps,
            tanimoto_threshold=0.3,
            mass_ppm_tolerance=50.0,
            random_seed=42,
        )

        # Audit counters
        stats = {
            "requested_negatives": 0,
            "sampled_negatives": 0,
            "requested_mass": 0,
            "sampled_mass": 0,
            "mass_shortfalls": 0,
            "requested_tanimoto": 0,
            "sampled_tanimoto": 0,
            "tanimoto_shortfalls": 0,
            "requested_random": 0,
            "sampled_random": 0,
            "random_shortfalls": 0,
            "identity_exclusions": 0,
            "duplicate_removals": 0,
            "anchors_with_0_negs": 0,
            "anchors_with_1_negs": 0,
            "anchors_with_2_negs": 0,
            "anchors_with_missing_adduct_mass": 0,
        }

        per_anchor_counts = []

        for idx_a in range(num_anchors):
            a_ikey = anchor_inchikeys[idx_a]
            a_fp = anchor_fps[idx_a]
            a_mz = anchor_mzs[idx_a]
            a_add = anchor_adducts[idx_a]
            a_mode = anchor_modes[idx_a]

            # Quota for this anchor
            k_mass = 1
            if idx_a % 2 == 0:
                k_tanimoto = 1
                k_random = 0
            else:
                k_tanimoto = 0
                k_random = 1

            req_total = k_mass + k_tanimoto + k_random
            stats["requested_negatives"] += req_total
            stats["requested_mass"] += k_mass
            stats["requested_tanimoto"] += k_tanimoto
            stats["requested_random"] += k_random

            # Check neutral mass validity
            n_mass = estimate_neutral_mass(a_mz, a_add, a_mode)
            if n_mass is None or n_mass <= 0:
                stats["anchors_with_missing_adduct_mass"] += 1

            # Count identity exclusions in pool
            id_in_pool = pool.inchikey_to_indices.get(a_ikey, [])
            stats["identity_exclusions"] += len(id_in_pool)

            # Sample negatives
            chosen_idxs, counts = pool.sample_negatives_for_anchor(
                anchor_inchikey=a_ikey,
                anchor_fp=a_fp,
                anchor_precursor_mz=a_mz,
                anchor_adduct=a_add,
                anchor_mode=a_mode,
                num_negatives=2,
                anchor_idx=idx_a,
                allow_adaptive_structural_fallback=True,
            )

            # Check duplicates
            unique_chosen = set(chosen_idxs)
            if len(unique_chosen) < len(chosen_idxs):
                stats["duplicate_removals"] += (len(chosen_idxs) - len(unique_chosen))

            # Record sampled counts
            stats["sampled_negatives"] += len(chosen_idxs)
            stats["sampled_mass"] += counts["mass_compatible"]
            stats["sampled_tanimoto"] += counts["tanimoto_hard"]
            stats["sampled_random"] += counts["random"]

            # Record shortfalls
            if counts["mass_compatible"] < k_mass:
                stats["mass_shortfalls"] += (k_mass - counts["mass_compatible"])
            if counts["tanimoto_hard"] < k_tanimoto:
                stats["tanimoto_shortfalls"] += (k_tanimoto - counts["tanimoto_hard"])
            if counts["random"] < k_random:
                stats["random_shortfalls"] += (k_random - counts["random"])

            # Per-anchor distribution
            n_tot = len(chosen_idxs)
            per_anchor_counts.append(n_tot)
            if n_tot == 0:
                stats["anchors_with_0_negs"] += 1
            elif n_tot == 1:
                stats["anchors_with_1_negs"] += 1
            elif n_tot == 2:
                stats["anchors_with_2_negs"] += 1

        # Summary Report
        tot_req = stats["requested_negatives"]
        tot_samp = stats["sampled_negatives"]
        print(f"Overall Negative Sampling Report ({num_anchors:,} anchors):")
        print(f"  Total Requested Negatives: {tot_req:,}")
        print(f"  Total Sampled Negatives:   {tot_samp:,} ({tot_samp / tot_req * 100:.2f}% fulfillment)")
        print(f"  Total Shortfall:           {tot_req - tot_samp:,}")
        print("\nCategory Fulfillment Breakdown:")
        print(f"  - Mass-Compatible: Requested {stats['requested_mass']:,} | Sampled {stats['sampled_mass']:,} | Shortfalls {stats['mass_shortfalls']:,} ({stats['sampled_mass']/stats['requested_mass']*100:.1f}%)")
        print(f"  - Morgan Tanimoto: Requested {stats['requested_tanimoto']:,} | Sampled {stats['sampled_tanimoto']:,} | Shortfalls {stats['tanimoto_shortfalls']:,} ({stats['sampled_tanimoto']/stats['requested_tanimoto']*100:.1f}%)")
        print(f"  - Random:          Requested {stats['requested_random']:,} | Sampled {stats['sampled_random']:,} | Shortfalls {stats['random_shortfalls']:,} ({stats['sampled_random']/stats['requested_random']*100:.1f}%)")

        print("\nActual Measured Category Proportions:")
        print(f"  - Mass-Compatible: {stats['sampled_mass']:,} / {tot_samp:,} ({stats['sampled_mass']/tot_samp*100:.2f}%)")
        print(f"  - Morgan Tanimoto: {stats['sampled_tanimoto']:,} / {tot_samp:,} ({stats['sampled_tanimoto']/tot_samp*100:.2f}%)")
        print(f"  - Random:          {stats['sampled_random']:,} / {tot_samp:,} ({stats['sampled_random']/tot_samp*100:.2f}%)")

        print("\nPer-Anchor Distribution:")
        print(f"  - Anchors with exactly 2 negatives: {stats['anchors_with_2_negs']:,} ({stats['anchors_with_2_negs']/num_anchors*100:.1f}%)")
        print(f"  - Anchors with exactly 1 negative:  {stats['anchors_with_1_negs']:,} ({stats['anchors_with_1_negs']/num_anchors*100:.1f}%)")
        print(f"  - Anchors with 0 negatives:          {stats['anchors_with_0_negs']:,} ({stats['anchors_with_0_negs']/num_anchors*100:.1f}%)")
        print(f"  - Mean negatives per anchor:         {np.mean(per_anchor_counts):.3f}")

        print("\nIntegrity Counters:")
        print(f"  - Identity Exclusions:               {stats['identity_exclusions']:,} instances masked")
        print(f"  - Duplicate Removals:                {stats['duplicate_removals']:,}")
        print(f"  - Anchors with Missing/Invalid Mass: {stats['anchors_with_missing_adduct_mass']:,}")


if __name__ == "__main__":
    run_sampler_audit(pool_sizes=[4348, 20000, 50000], num_anchors=1000)

