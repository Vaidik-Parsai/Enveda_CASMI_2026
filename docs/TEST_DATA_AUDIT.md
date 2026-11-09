# Enveda CASMI 2026: Competition Test Dataset Audit

**Document ID:** `docs/TEST_DATA_AUDIT.md`  
**Phase:** Phase 5 — Competition Test Data Exploration & Schema Audit  
**Auditor:** Senior ML Research Engineer (Mass Spectrometry & Systems)  

---

## 1. Test Dataset Location & Overview

* **File Location:** `C:\Users\vaidi\.cache\kagglehub\competitions\enveda-CASMI26-molecule-id-mass-spectra\test.parquet`
* **Sample Submission File:** `C:\Users\vaidi\.cache\kagglehub\competitions\enveda-CASMI26-molecule-id-mass-spectra\sample_submission.csv`
* **Total Records:** **1,213** experimental MS/MS spectra.
* **Target Query Molecules:** Exactly **400** unique `molecule_id` values.
* **Ground-Truth Labels:** **0** (strictly unlabelled test set; `normalized_smiles` and `inchikey14` are completely absent).

---

## 2. Schema Comparison: Training vs. Test

| Field Name | Type in `train.parquet` | Type in `test.parquet` | Missing / Null in Test | Notes |
|---|---|---|:---:|---|
| `molecule_id` | `str` (or hashed) | `str` | 0 (0.0%) | Identifier for grouping query spectra (`m_005e53`, etc.) |
| `spectrum_id` | `str` | `str` | 0 (0.0%) | Unique identifier per acquisition |
| `ms2_mzs` | `list<float64>` | `list<float64>` | 0 (0.0%) | Fragment ion mass-to-charge values |
| `ms2_normalized_intensities` | `list<float64>` | `list<float64>` | 0 (0.0%) | Normalized relative fragment ion intensities |
| `base_peak_intensity` | `float64` | `float64` | 0 (0.0%) | Absolute intensity of base fragment |
| `adduct` | `str` | `str` | 0 (0.0%) | Adduct ion formula (`[M+H]+`, `[M-H]-`, etc.) |
| `ionization_mode` | `str` | `str` | 0 (0.0%) | Polarity (`positive`, `negative`) |
| `instrument_type` | `str` | `str` | 0 (0.0%) | Instrument category (`HCD`, `Orbitrap`, etc.) |
| `precursor_mz` | `float64` | `float64` | 0 (0.0%) | Observed intact precursor $m/z$ |
| `collision_energy_orig` | `str` | `str` | 0 (0.0%) | Raw acquisition collision energy notation |
| `collision_energy_ev` | `object` / `float` | `object` / `float` | 0 (0.0%) | Parsed collision energy in electron-volts |
| `collision_energy_orig_units` | `str` | `str` | 0 (0.0%) | Nominal units (`eV`, `NCE`, `%`) |
| `normalized_smiles` | `str` | **Absent** | — | Ground truth withheld by competition organizers |
| `inchikey14` | `str` | **Absent** | — | Ground truth connectivity withheld |

**Completeness:** Across all 1,213 test spectra and all 12 test columns, there are **exactly zero missing values**.

---

## 3. Query Molecule & Spectral Statistics

### 3.1 Spectrum Multiplicity per Query Molecule
In the test set, each molecule is represented by multiple spectra acquired under differing collision energies and ionization modes:
* **Total Query Molecules:** 400
* **Mean Spectra per Molecule:** 3.0325
* **Standard Deviation:** 1.3825
* **Minimum Spectra:** 1 spectrum
* **25th Percentile:** 2 spectra
* **Median (50th Percentile):** 3 spectra
* **75th Percentile:** 4 spectra
* **Maximum Spectra:** 9 spectra

### 3.2 Adduct Distribution in Test Data
Every adduct in the test set is accounted for in `src/enveda/evaluation/mass_filter.py:ADDUCT_RULES`:

| Adduct Species | Ionization Mode | Test Spectrum Count | Proportion | Exact Neutral Shift ($\Delta m$) |
|---|:---:|---:|---:|:---:|
| `[M+H]+` | Positive | 959 | 79.06% | $+1.007276\text{ Da}$ |
| `[M-H]-` | Negative | 193 | 15.91% | $-1.007276\text{ Da}$ |
| `[M+CH2O2-H]-` (Formate) | Negative | 31 | 2.56% | $+44.998201\text{ Da}$ |
| `[M+Na]+` | Positive | 22 | 1.81% | $+22.989218\text{ Da}$ |
| `[M+NH4]+` | Positive | 4 | 0.33% | $+18.033823\text{ Da}$ |
| `[M+K]+` | Positive | 2 | 0.16% | $+38.963158\text{ Da}$ |
| `[M+Cl]-` | Negative | 2 | 0.16% | $+34.969402\text{ Da}$ |

### 3.3 Precursor $m/z$ Range
* **Minimum $m/z$:** $245.0931\text{ Da}$
* **25th Percentile:** $300.1384\text{ Da}$
* **Median:** $329.1978\text{ Da}$
* **Mean:** $327.9089\text{ Da}$
* **75th Percentile:** $347.1611\text{ Da}$
* **Maximum $m/z$:** $460.1554\text{ Da}$

The mass distribution is concentrated in the typical drug-like natural product / small-molecule range ($250\text{–}450\text{ Da}$).

---

## 4. Required Submission Format
From `sample_submission.csv`:
* **File Format:** CSV with header `molecule_id,smiles`.
* **Row Count:** Exactly 400 rows (one per unique `molecule_id`).
* **SMILES Column:** Up to 25 ranked candidate SMILES strings joined by semicolon delimiter (`;`), ordered from highest confidence (rank 1) to lowest (rank 25).
* Example:
  ```csv
  molecule_id,smiles
  m_005e53,CCO;CCCCO;c1ccccc1...
  m_006153,CC(=O)O;...
  ```
