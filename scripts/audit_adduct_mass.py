"""Audit script for Adduct and Mass Validation (Section B)."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enveda.config import load_config
from enveda.data import create_or_load_group_split
from enveda.evaluation.mass_filter import (
    ADDUCT_RULES,
    calculate_exact_monoisotopic_mass,
    estimate_neutral_mass,
    get_molecule_consensus_neutral_mass,
)


def run_adduct_audit():
    config = load_config("configs/default.yaml")
    train_path = Path(config.data.train_path)
    test_path = Path(config.data.test_path)
    split_path = Path(config.data.split_file)

    print("=" * 80)
    print("SECTION B: ADDUCT AND MASS VALIDATION AUDIT")
    print("=" * 80)

    # 1. RDKit exact mass calculation test on reference standards
    test_standards = [
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "formula": "C8H10N4O2", "expected_monoisotopic": 194.0803756},
        {"name": "D-Glucose", "smiles": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O", "formula": "C6H12O6", "expected_monoisotopic": 180.0633881},
        {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "formula": "C9H8O4", "expected_monoisotopic": 180.0422587},
        {"name": "ATP", "smiles": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O", "formula": "C10H16N5O13P3", "expected_monoisotopic": 506.995745},
    ]

    print("\n--- 1. Reference Standards Mass Validation (RDKit) ---")
    for std in test_standards:
        calc_mass = calculate_exact_monoisotopic_mass(std["smiles"])
        diff_da = abs(calc_mass - std["expected_monoisotopic"])
        diff_ppm = (diff_da / std["expected_monoisotopic"]) * 1e6
        print(f"  {std['name']:<12} | SMILES: {std['smiles'][:25]}... | Calc: {calc_mass:.6f} Da | Ref: {std['expected_monoisotopic']:.6f} Da | Error: {diff_ppm:.3f} ppm (Pass: {diff_ppm < 1.0})")
        assert diff_ppm < 1.0, f"Mass calculation error exceeds 1 ppm for {std['name']}"

    # 2. Adduct definitions and offset tests
    print("\n--- 2. Adduct Definitions and Charge-State Offsets ---")
    sample_neutral = 194.080376  # Caffeine
    for adduct, meta in ADDUCT_RULES.items():
        delta = meta["mass_shift"]
        charge = abs(meta["charge"])
        mult = meta["multiplier"]
        mode_str = "positive" if meta["charge"] > 0 else "negative"
        # Simulated precursor m/z: mz = (M * mult + delta) / charge
        sim_mz = (sample_neutral * mult + delta) / charge
        # Reconstruct neutral
        reconstructed = estimate_neutral_mass(sim_mz, adduct, mode_str)
        err_ppm = (abs(reconstructed - sample_neutral) / sample_neutral) * 1e6
        print(f"  Adduct: {adduct:<15} | Charge: {charge:.0f} | Multiplier: {mult:.0f} | Shift: {delta:10.6f} Da | Sim m/z: {sim_mz:10.4f} | Recon Error: {err_ppm:.6f} ppm")
        assert err_ppm < 1e-4, f"Reconstruction failed for adduct {adduct}"

    # 3. Adduct distribution in Train Split and Test Set
    print("\n--- 3. Dataset Adduct Distributions & Missing/Unknown Frequency ---")
    _, val_rows = create_or_load_group_split(train_path, split_path)

    val_table = pq.read_table(str(train_path), columns=["adduct", "ionization_mode", "precursor_mz", "normalized_smiles", "inchikey14"])
    val_df_full = val_table.to_pandas().iloc[val_rows].reset_index(drop=True)
    val_df = val_df_full.iloc[:17267]  # 400 validation molecules

    test_table = pq.read_table(str(test_path), columns=["adduct", "ionization_mode", "precursor_mz", "molecule_id"])
    test_df = test_table.to_pandas()

    print(f"\nValidation Split ({len(val_df):,} spectra, 400 molecules):")
    val_adduct_counts = val_df["adduct"].value_counts(dropna=False)
    for add, count in val_adduct_counts.items():
        pct = (count / len(val_df)) * 100
        known = add in ADDUCT_RULES
        print(f"  {str(add):<20} : {count:6d} ({pct:5.2f}%) | Known Supported: {known}")

    print(f"\nTest Set ({len(test_df):,} spectra, {test_df['molecule_id'].nunique():,} molecules):")
    test_adduct_counts = test_df["adduct"].value_counts(dropna=False)
    for add, count in test_adduct_counts.items():
        pct = (count / len(test_df)) * 100
        known = add in ADDUCT_RULES
        print(f"  {str(add):<20} : {count:6d} ({pct:5.2f}%) | Known Supported: {known}")

    val_unknown_count = sum(count for add, count in val_adduct_counts.items() if add not in ADDUCT_RULES)
    test_unknown_count = sum(count for add, count in test_adduct_counts.items() if add not in ADDUCT_RULES)
    print(f"\nUnknown / Missing Adducts: Validation = {val_unknown_count} ({(val_unknown_count/len(val_df))*100:.2f}%), Test = {test_unknown_count} ({(test_unknown_count/len(test_df))*100:.2f}%)")

    # 4. Precursor Mass Error Distribution in ppm
    print("\n--- 4. Precursor Mass Error Distribution (Validation Spectra vs True Neutral Mass) ---")
    ppm_errors = []
    abs_ppm_errors = []

    for idx, row in val_df.iterrows():
        smi = row["normalized_smiles"]
        true_mass = calculate_exact_monoisotopic_mass(smi)
        pmz = row["precursor_mz"]
        add = row["adduct"]
        mode = row["ionization_mode"]

        est_mass = estimate_neutral_mass(pmz, add, mode)
        if est_mass is not None:
            err_ppm = ((est_mass - true_mass) / true_mass) * 1e6
            ppm_errors.append(err_ppm)
            abs_ppm_errors.append(abs(err_ppm))

    ppm_errors = np.array(ppm_errors)
    abs_ppm_errors = np.array(abs_ppm_errors)

    print(f"Total evaluated spectra: {len(ppm_errors):,}")
    print(f"Mean ppm error:       {np.mean(ppm_errors):+.2f} ppm")
    print(f"Median ppm error:     {np.median(ppm_errors):+.2f} ppm")
    print(f"Std dev ppm error:    {np.std(ppm_errors):.2f} ppm")
    print(f"Mean |ppm| error:     {np.mean(abs_ppm_errors):.2f} ppm")
    print(f"Median |ppm| error:   {np.median(abs_ppm_errors):.2f} ppm")
    print(f"Percentiles |ppm|:")
    print(f"  50th percentile:    {np.percentile(abs_ppm_errors, 50):.2f} ppm")
    print(f"  90th percentile:    {np.percentile(abs_ppm_errors, 90):.2f} ppm")
    print(f"  95th percentile:    {np.percentile(abs_ppm_errors, 95):.2f} ppm")
    print(f"  99th percentile:    {np.percentile(abs_ppm_errors, 99):.2f} ppm")
    print(f"  Max |ppm| error:    {np.max(abs_ppm_errors):.2f} ppm")

    pct_within_5ppm = np.mean(abs_ppm_errors <= 5.0) * 100
    pct_within_10ppm = np.mean(abs_ppm_errors <= 10.0) * 100
    pct_within_20ppm = np.mean(abs_ppm_errors <= 20.0) * 100
    pct_within_30ppm = np.mean(abs_ppm_errors <= 30.0) * 100
    pct_within_50ppm = np.mean(abs_ppm_errors <= 50.0) * 100

    print(f"\nMass Accuracy Coverage:")
    print(f"  Within 5 ppm:   {pct_within_5ppm:.2f}%")
    print(f"  Within 10 ppm:  {pct_within_10ppm:.2f}%")
    print(f"  Within 20 ppm:  {pct_within_20ppm:.2f}%")
    print(f"  Within 30 ppm:  {pct_within_30ppm:.2f}%")
    print(f"  Within 50 ppm:  {pct_within_50ppm:.2f}%")

    # 5. Multi-spectrum consensus neutral mass consistency per molecule
    print("\n--- 5. Consensus Neutral Mass Aggregation Across Multiple Spectra ---")
    mol_groups = val_df.groupby("inchikey14", sort=False)
    consensus_ppm_errors = []
    spectra_per_mol = []

    for m_id, group in mol_groups:
        true_smi = group["normalized_smiles"].iloc[0]
        true_mass = calculate_exact_monoisotopic_mass(true_smi)
        pmzs = group["precursor_mz"].tolist()
        adds = group["adduct"].tolist()
        modes = group["ionization_mode"].tolist()
        c_mass = get_molecule_consensus_neutral_mass(pmzs, adds, modes)

        c_err = abs((c_mass - true_mass) / true_mass) * 1e6
        consensus_ppm_errors.append(c_err)
        spectra_per_mol.append(len(group))

    consensus_ppm_errors = np.array(consensus_ppm_errors)
    print(f"Evaluated molecules: {len(consensus_ppm_errors)}")
    print(f"Mean spectra per molecule:   {np.mean(spectra_per_mol):.1f} (min: {min(spectra_per_mol)}, max: {max(spectra_per_mol)})")
    print(f"Consensus Mean |ppm| error:  {np.mean(consensus_ppm_errors):.2f} ppm")
    print(f"Consensus Median |ppm| error:{np.median(consensus_ppm_errors):.2f} ppm")
    print(f"Consensus 95th percentile:   {np.percentile(consensus_ppm_errors, 95):.2f} ppm")
    print(f"Consensus 99th percentile:   {np.percentile(consensus_ppm_errors, 99):.2f} ppm")
    print(f"Consensus within 10 ppm:     {np.mean(consensus_ppm_errors <= 10.0)*100:.2f}%")
    print(f"Consensus within 20 ppm:     {np.mean(consensus_ppm_errors <= 20.0)*100:.2f}%")
    print(f"Consensus within 30 ppm:     {np.mean(consensus_ppm_errors <= 30.0)*100:.2f}%")

    # 4. Precursor Mass Error Distribution in ppm
    print("\n--- 4. Precursor Mass Error Distribution (Validation Spectra vs True Neutral Mass) ---")
    ppm_errors = []
    abs_ppm_errors = []
    skipped_count = 0

    for idx, row in val_df.iterrows():
        smi = row["normalized_smiles"]
        true_mass = calculate_exact_monoisotopic_mass(smi)
        pmz = row["precursor_mz"]
        add = row["adduct"]
        mode = row["ionization_mode"]

        est_mass = estimate_neutral_mass_from_precursor(pmz, add, mode)
        if est_mass is not None:
            err_ppm = ((est_mass - true_mass) / true_mass) * 1e6
            ppm_errors.append(err_ppm)
            abs_ppm_errors.append(abs(err_ppm))
        else:
            skipped_count.append(idx)

    ppm_errors = np.array(ppm_errors)
    abs_ppm_errors = np.array(abs_ppm_errors)

    print(f"Total evaluated spectra: {len(ppm_errors):,}")
    print(f"Mean ppm error:       {np.mean(ppm_errors):+.2f} ppm")
    print(f"Median ppm error:     {np.median(ppm_errors):+.2f} ppm")
    print(f"Std dev ppm error:    {np.std(ppm_errors):.2f} ppm")
    print(f"Mean |ppm| error:     {np.mean(abs_ppm_errors):.2f} ppm")
    print(f"Median |ppm| error:   {np.median(abs_ppm_errors):.2f} ppm")
    print(f"Percentiles |ppm|:")
    print(f"  50th percentile:    {np.percentile(abs_ppm_errors, 50):.2f} ppm")
    print(f"  90th percentile:    {np.percentile(abs_ppm_errors, 90):.2f} ppm")
    print(f"  95th percentile:    {np.percentile(abs_ppm_errors, 95):.2f} ppm")
    print(f"  99th percentile:    {np.percentile(abs_ppm_errors, 99):.2f} ppm")
    print(f"  Max |ppm| error:    {np.max(abs_ppm_errors):.2f} ppm")

    pct_within_5ppm = np.mean(abs_ppm_errors <= 5.0) * 100
    pct_within_10ppm = np.mean(abs_ppm_errors <= 10.0) * 100
    pct_within_20ppm = np.mean(abs_ppm_errors <= 20.0) * 100
    pct_within_30ppm = np.mean(abs_ppm_errors <= 30.0) * 100
    pct_within_50ppm = np.mean(abs_ppm_errors <= 50.0) * 100

    print(f"\nMass Accuracy Coverage:")
    print(f"  Within 5 ppm:   {pct_within_5ppm:.2f}%")
    print(f"  Within 10 ppm:  {pct_within_10ppm:.2f}%")
    print(f"  Within 20 ppm:  {pct_within_20ppm:.2f}%")
    print(f"  Within 30 ppm:  {pct_within_30ppm:.2f}%")
    print(f"  Within 50 ppm:  {pct_within_50ppm:.2f}%")

    # 5. Multi-spectrum consensus neutral mass consistency per molecule
    print("\n--- 5. Consensus Neutral Mass Aggregation Across Multiple Spectra ---")
    mol_groups = val_df.groupby("inchikey14", sort=False)
    consensus_ppm_errors = []
    spectra_per_mol = []

    for m_id, group in mol_groups:
        true_smi = group["normalized_smiles"].iloc[0]
        true_mass = calculate_exact_monoisotopic_mass(true_smi)
        pmzs = group["precursor_mz"].tolist()
        adds = group["adduct"].tolist()
        modes = group["ionization_mode"].tolist()
        c_mass = get_molecule_consensus_neutral_mass(pmzs, adds, modes)

        c_err = abs((c_mass - true_mass) / true_mass) * 1e6
        consensus_ppm_errors.append(c_err)
        spectra_per_mol.append(len(group))

    consensus_ppm_errors = np.array(consensus_ppm_errors)
    print(f"Evaluated molecules: {len(consensus_ppm_errors)}")
    print(f"Mean spectra per molecule:   {np.mean(spectra_per_mol):.1f} (min: {min(spectra_per_mol)}, max: {max(spectra_per_mol)})")
    print(f"Consensus Mean |ppm| error:  {np.mean(consensus_ppm_errors):.2f} ppm")
    print(f"Consensus Median |ppm| error:{np.median(consensus_ppm_errors):.2f} ppm")
    print(f"Consensus 95th percentile:   {np.percentile(consensus_ppm_errors, 95):.2f} ppm")
    print(f"Consensus 99th percentile:   {np.percentile(consensus_ppm_errors, 99):.2f} ppm")
    print(f"Consensus within 10 ppm:     {np.mean(consensus_ppm_errors <= 10.0)*100:.2f}%")
    print(f"Consensus within 20 ppm:     {np.mean(consensus_ppm_errors <= 20.0)*100:.2f}%")
    print(f"Consensus within 30 ppm:     {np.mean(consensus_ppm_errors <= 30.0)*100:.2f}%")

    # Save summary stats
    audit_data = {
        "val_adduct_counts": {str(k): int(v) for k, v in val_adduct_counts.items()},
        "test_adduct_counts": {str(k): int(v) for k, v in test_adduct_counts.items()},
        "mean_ppm_error": float(np.mean(ppm_errors)),
        "std_ppm_error": float(np.std(ppm_errors)),
        "median_abs_ppm_error": float(np.median(abs_ppm_errors)),
        "p95_abs_ppm_error": float(np.percentile(abs_ppm_errors, 95)),
        "pct_within_20ppm": float(pct_within_20ppm),
        "pct_within_30ppm": float(pct_within_30ppm),
        "consensus_median_abs_ppm": float(np.median(consensus_ppm_errors)),
        "consensus_within_20ppm": float(np.mean(consensus_ppm_errors <= 20.0)*100),
    }

    with open("outputs/adduct_mass_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print("\nSaved adduct and mass audit data to outputs/adduct_mass_audit.json")


if __name__ == "__main__":
    run_adduct_audit()
