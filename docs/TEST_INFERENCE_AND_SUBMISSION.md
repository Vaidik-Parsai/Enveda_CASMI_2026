# Enveda CASMI 2026: Competition Test Inference & Submission Report

**Document ID:** `docs/TEST_INFERENCE_AND_SUBMISSION.md`  
**Phase:** Phase 6 — Test Inference Execution & Submission Integrity Verification  
**Auditor:** Senior ML Research Engineer (Mass Spectrometry & Systems)  
**Execution Timestamp:** 2026-11-09T17:12:36+05:30  

---

## 1. Execution Summary

Competition test inference was executed exclusively using the official champion model checkpoint:
* **Model Checkpoint:** `checkpoints/spectrum_molecule_step_5000.pt` (Verified intact)
* **Candidate Bank Universe:** 277,566 unique molecular structures (`outputs/candidate_bank/`)
* **Scoring Protocol:** Precursor Consensus Neutral Mass Estimation with Gaussian Soft Mass Compatibility ($\sigma = 20.0\text{ ppm}$) and Multi-Spectrum Molecule-Level Mean Aggregation (`MASS_MEAN`).
* **Test Spectra Processed:** 1,213 experimental MS/MS spectra from `test.parquet`.
* **Output Predictions:** 400 query molecules, each receiving exactly 25 ranked candidate SMILES strings (total 10,000 predictions).

---

## 2. Test File & Preprocessing Provenance

* **Input Test File:** `C:\Users\vaidi\.cache\kagglehub\competitions\enveda-CASMI26-molecule-id-mass-spectra\test.parquet`
* **Official Template:** `C:\Users\vaidi\.cache\kagglehub\competitions\enveda-CASMI26-molecule-id-mass-spectra\sample_submission.csv`
* **Metadata Preprocessor:** `data/cache/meta_preprocessor.joblib` (130-dimensional z-score scaling and categorical one-hot encoding).
* **Spectrum Preprocessor:** Relative noise threshold $\ge 0.01$, top-128 peak selection, 10,000 uniform bins ($\Delta m/z = 0.1\text{ Da}$) spanning $[0.0, 1000.0\text{ Da}]$.
* **Inference Guardrails:**
  - Zero target structures or ground-truth labels were used (test set is strictly unlabelled).
  - Precursor neutral mass estimation was performed blindly using observed precursor $m/z$, charge, and adduct formulas.

---

## 3. Submission File Schema & Verification

### 3.1 Primary Submission Artifacts
* **Primary Submission File:** `outputs/submission_top25.csv`
* **Mirrored Submission File:** `outputs/submission.csv`

### 3.2 Verification & Integrity Checklist

| Integrity Requirement | Expected Value | Measured / Verified Value | Pass / Fail |
|---|---|---|:---:|
| **Output File Header** | `molecule_id,smiles` | `molecule_id,smiles` | **PASS** |
| **Total Query Molecules** | Exactly 400 rows | 400 rows | **PASS** |
| **Unique Molecule IDs** | 400 unique IDs | 400 unique IDs | **PASS** |
| **ID Alignment with Sample** | 100% exact match | Identical IDs and ordering | **PASS** |
| **Delimiter** | Semicolon (`;`) | Semicolon (`;`) | **PASS** |
| **Candidates per Molecule** | Exactly 25 ranked SMILES | Exactly 25 for all 400 rows | **PASS** |
| **Total Ranked Predictions** | 10,000 predictions | 10,000 predictions | **PASS** |
| **Candidate Bank Origin** | 100% from 277,566 library | 10,000 / 10,000 in candidate library | **PASS** |
| **Within-Row Deduplication** | 0 duplicate SMILES per row | 0 duplicate candidates detected | **PASS** |
| **Missing Values / NaNs** | 0 nulls | 0 nulls, 0 NaNs, 0 Infs | **PASS** |
| **Chemical Validity (RDKit)** | Valid 2D structures | 100% parseable canonical SMILES | **PASS** |

---

## 4. Leaderboard Reporting Policy & Scientific Integrity Notice

> [!IMPORTANT]
> **Ground-Truth Withheld by Organizers:**
> Because ground-truth chemical structures for `test.parquet` are withheld by Enveda Biosciences and the CASMI competition organizers, **no test accuracy or test MRR@25 is reported here**. Claiming a test accuracy or MRR without access to ground truth would be scientifically fraudulent.
> 
> When this submission file is uploaded to the official competition evaluation platform, the returned leaderboard MRR@25 score should be recorded in this document as the official external competition evaluation metric.

---

## 5. Submission Execution Command
The final submission was generated using:
```powershell
.\venv\Scripts\python scripts/predict.py `
    --checkpoint checkpoints/spectrum_molecule_step_5000.pt `
    --candidate_dir outputs/candidate_bank `
    --method mass_mean `
    --sigma_ppm 20.0 `
    --top_k 25 `
    --output_csv outputs/submission_top25.csv
```
