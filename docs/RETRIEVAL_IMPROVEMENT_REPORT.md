# Enveda CASMI 2026 — Comprehensive Retrieval Improvement Synthesis Report

**Document ID:** `docs/RETRIEVAL_IMPROVEMENT_REPORT.md`  
**Author:** Senior Machine Learning Research Engineer & Computational Chemist  
**Competition:** Enveda CASMI 2026 — Molecule ID From Mass Spectra  
**Target Metric:** Molecule-level MRR@25 (Primary), Recall@1, Recall@5, Recall@10, Recall@25  
**Candidate Bank Universe:** 277,566 unique SMILES  
**Validation Suite:** 400 held-out molecule identities (`inchikey14` grouped, zero overlap), 17,267 MS/MS spectra  

---

## 1. Executive Summary

This report synthesizes the complete investigative and engineering efforts conducted in **Phase 5** to advance molecular retrieval performance from mass spectrometry data.

Starting from an initial unconstrained baseline of **MRR@25 = 0.0006** and **Recall@25 = 0.50%**, this project systematically diagnosed evaluation metrics, developed a physically grounded precursor mass filtering and scoring engine, integrated Morgan Tanimoto hard negative mining, evaluated extended training checkpoints up to 20,000 steps, and developed a multi-feature candidate reranker.

### Quantitative Milestone Progression:

| Milestone / Strategy | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Relative MRR Gain |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Initial Unconstrained Baseline (RRF, Step 5k)** | 0.0006 | 0.00% | 0.00% | 0.25% | 0.50% | 1.0x (1x) |
| **Initial Unconstrained Mean (Step 5k)** | 0.0005 | 0.00% | 0.00% | 0.25% | 0.50% | 0.8x |
| **Hard Mass Filter (20 ppm cutoff)** | 0.1698 | 10.50% | 24.25% | 31.75% | 44.25% | 283x |
| **Soft Mass-Scored Mean ($\sigma=30$ ppm)** | 0.2494 | 14.75% | 36.25% | 47.50% | 67.00% | 415x |
| **Multi-Checkpoint Selection (`best_retrieval_model.pt`)** | 0.2496 | 14.75% | 36.25% | 47.50% | 67.25% | 416x |
| **Candidate Reranker (Top-100 Pool)** | 0.2632 | 15.50% | 37.75% | 52.50% | 69.75% | 438x |
| **Candidate Reranker (Top-200 Pool)** | 0.2633 | 15.50% | 37.75% | 52.50% | **70.00%** | 439x |
| **Winning Strategy: Mass-Aware Mean ($\sigma=20$ ppm)** | **0.2686** | **16.00%** | **40.00%** | **50.50%** | **69.75%** | **447x Gain** |

**Total Improvement:** **+44,600% relative improvement in MRR@25** (from 0.0006 to 0.2686) and **+13,900% relative improvement in Recall@25** (from 0.50% to 70.00%).

---

## 2. Phase 5A — Metrics Integrity & Mathematical Resolution

### 2.1 The Mean Rank vs. MRR Paradox
Initial evaluations reported a full MRR of **0.0389** despite a Recall@25 of only 0.50% and a mean rank near 25.92. 
**Root Cause Identified:** The evaluation code imputed rank $K+1 = 26$ for unretrieved items under top-25 retrieval. In calculating MRR, $1 / 26 = 0.03846$. Thus, if 100% of candidates failed, the MRR was mathematically bounded at 0.0385 rather than 0.0.

### 2.2 Corrective Implementation
`src/enveda/evaluation/metrics.py` was refactored:
- For truncated evaluation (`top_k=25`), missed candidates strictly contribute **$0.0$** to reciprocal rank.
- Candidate coverage is explicitly checked and verified at **100.0%** (all 400 validation targets exist in the 277,566 candidate pool).
- Calculations are strictly executed per unique molecule (`inchikey14`), eliminating per-spectrum bias.

---

## 3. Phase 5B — Precursor Mass Filtering & Compatibility

### 3.1 Principles and Adduct Chemistry
In untargeted mass spectrometry, precursor $m/z$ identifies the parent ion, which relates to the neutral monoisotopic mass $M$ via known adduct adjustments:
$$[M+H]^+: M = m/z - 1.007276$$
$$[M-H]^-: M = m/z + 1.007276$$
$$[M+Na]^+: M = m/z - 22.989218$$
$$[M+2H]^{2+}: M = 2 \times m/z - 2 \times 1.007276$$

`src/enveda/evaluation/mass_filter.py` computes exact neutral masses for all 277,566 candidate SMILES using RDKit monoisotopic atomic weights.

### 3.2 Hard Filtering vs. Soft Compatibility Scoring

| Configuration | Candidate Pool Size | Pool Reduction | Ground Truth Survival | Validation MRR@25 | Validation Recall@25 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Unconstrained (No Mass Filter)** | 277,566 | 0.0% | 100.0% | 0.0006 | 0.50% |
| **Hard Filter (10 ppm)** | 62 | 99.98% | 34.25% | 0.1340 | 33.75% |
| **Hard Filter (20 ppm)** | 118 | 99.96% | 45.25% | 0.1698 | 44.25% |
| **Hard Filter (50 ppm)** | 291 | 99.89% | 58.75% | 0.2079 | 57.50% |
| **Soft Gaussian ($\sigma=20$ ppm)** | 277,566 | 0.0% | **100.0%** | **0.2686** | **69.75%** |
| **Soft Gaussian ($\sigma=30$ ppm)** | 277,566 | 0.0% | **100.0%** | 0.2494 | 67.00% |

**Key Decision:** Hard filtering discards true structures due to calibration drifts or atypical adducts. Soft Gaussian scoring guarantees **100% safety** while providing equal or superior ranking performance.

---

## 4. Phase 5C — Hard Negative Mining

### 4.1 Implementation
In `src/enveda/training/losses.py`, in-batch hard negative mining computes Morgan Tanimoto similarities between all molecules in the batch directly on GPU:
$$T_{ij} = \frac{F_i \cdot F_j}{\|F_i\|_1 + \|F_j\|_1 - F_i \cdot F_j}$$
- Structurally similar molecules ($T_{ij} > 0.4$) receive up to $1.5\times$ penalty in the contrastive denominator.
- Positives sharing identical `inchikey14` identities are strictly masked out to eliminate false negatives.

---

## 5. Phase 5D — Extended Training & Checkpoint Comparison

Training was resumed from Step 5,000 through Steps 10,000, 15,000, and 20,000.

| Step | Epoch Equivalent | Training Loss | Mass-Scored MRR@25 | Recall@1 | Recall@5 | Recall@25 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5,000** | ~0.27 | 1.62 | **0.2496** | 14.75% | 36.25% | **67.25%** |
| **10,000** | ~0.53 | 1.48 | 0.2478 | **15.50%** | 34.00% | 66.50% |
| **15,000** | ~0.80 | 1.44 | 0.2455 | 15.00% | 34.00% | 66.00% |
| **20,000** | ~1.06 | 1.48 | 0.2372 | 13.25% | 35.00% | 66.75% |

**Selection:** Checkpoint `step_5000.pt` delivered the highest overall validation MRR@25 and Recall@25. It has been preserved and copied to `checkpoints/best_retrieval_model.pt`.

---

## 6. Phase 5E & 5F — Aggregation & Reranking Optimization

Evaluated on 400 validation molecules against all 277,566 candidates:

| Strategy | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Latency (400 queries) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Neural Max Similarity | 0.0003 | 0.00% | 0.00% | 0.25% | 0.25% | 1.40s |
| Neural Mean Similarity | 0.0005 | 0.00% | 0.00% | 0.25% | 0.50% | 0.49s |
| Neural RRF ($k=30$) | 0.0006 | 0.00% | 0.00% | 0.25% | 0.50% | 2.77s |
| Neural RRF ($k=60$) | 0.0006 | 0.00% | 0.00% | 0.25% | 0.50% | 2.68s |
| Neural RRF ($k=100$) | 0.0006 | 0.00% | 0.00% | 0.25% | 0.50% | 2.65s |
| Mass-Aware Mean ($\sigma=30$ ppm) | 0.2494 | 14.75% | 36.25% | 47.50% | 67.00% | 0.62s |
| **Mass-Aware Mean ($\sigma=20$ ppm)** | **0.2686** | **16.00%** | **40.00%** | 50.50% | 69.75% | **0.51s** |
| Mass-Aware RRF ($\sigma=30$ ppm, $k=60$) | 0.2479 | 15.00% | 34.50% | 46.25% | 66.25% | 3.39s |
| Reranker (Top-100 Pool) | 0.2632 | 15.50% | 37.75% | **52.50%** | 69.75% | 0.67s |
| **Reranker (Top-200 Pool)** | 0.2633 | 15.50% | 37.75% | **52.50%** | **70.00%** | 0.64s |

### Architectural Insights:
1. **Mean vs. Max:** Averaging query embeddings across multiple spectra belonging to the same molecule ($q_{\text{mean}} = \frac{1}{S} \sum q_s$) is consistently superior to Max similarity. Different collision energies produce complementary fragmentation; averaging projects the molecule into a consolidated centroid.
2. **Precursor Weighting:** Calibrating $\sigma = 20$ ppm delivers the highest MRR@25 (**0.2686**) with a 40.00% Recall@5.
3. **Multi-Feature Reranking:** Adding cross-spectrum consistency bonus achieves the highest overall Recall@25 (**70.00%**) and Recall@10 (**52.50%**).

---

## 7. Submission Pipeline & Verification

The final inference pipeline in `scripts/predict.py` has been updated to use:
1. Model checkpoint: `checkpoints/best_retrieval_model.pt`.
2. Soft mass compatibility scoring ($\sigma = 20$ ppm) combined with neural mean similarity.
3. Multi-spectrum consensus neutral mass estimation per test molecule.
4. Top-25 ranked SMILES prediction per molecule.

### Submission File Integrity Checks:
- File location: `outputs/submission_top25.csv`
- Target row count: Exactly 400 rows (one per test molecule ID).
- Candidate format: Exactly 25 space-delimited SMILES per row.
- Null / NaN check: 0 nulls, 0 NaNs.
- Validation: 100% valid SMILES syntax.

