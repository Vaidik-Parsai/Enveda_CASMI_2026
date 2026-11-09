# Enveda CASMI 2026: Experimental Progression & Ablation Studies

**Document ID:** `docs/EXPERIMENTS.md`  
**Purpose:** Comprehensive experimental history, empirical benchmarks, ablation studies, and architectural decisions leading to the final champion model.  

---

## 1. Experimental Progression Overview

The development trajectory of the Enveda CASMI 2026 retrieval system progressed across six rigorous experimental phases:

```
[Phase 1-4: Raw Dual-Encoder Baseline]
        │   MRR@25: 0.0006 | Recall@25: 0.50%
        ▼
[Phase 5: Mass-Aware Soft Scoring Formulation]
        │   MRR@25: 0.2686 | Recall@25: 69.75%  (+44,667% relative gain)
        ▼
[Phase 5 Ablations: Mass vs Neural & Sigma Optimization]
        │   Identified sigma = 20 ppm as optimal balance
        ▼
[Phase 6A: Extended Training Audit & Failure Mode Diagnosis]
        │   Diagnosed dimensional collapse, in-batch negative starvation & constant LR drift
        ▼
[Phase 6B Run A: Cosine Warmup Learning-Rate Annealing]
        │   MRR@25: 0.2573 | Recall@25: 67.75%
        ▼
[Phase 6B Run C: Chemistry-Aware Hard-Negative Sampling]
        │   MRR@25: 0.2525 | Recall@25: 70.50%  (All-Time Record Recall)
        ▼
[Final Champion Decision: Preserve Step 5,000 Baseline (MRR@25: 0.2686)]
```

---

## 2. Phase 5: The Mass-Aware Scoring Breakthrough

### 2.1 The Baseline Failure Mode
In initial evaluations (Phase 1–4), the dual-encoder model relied solely on raw neural cosine similarity between spectrum embeddings and candidate molecule embeddings:
$$S(s, c) = \langle z_{\text{spec}}^{(s)}, z_{\text{mol}}^{(c)} \rangle$$

Across 400 held-out validation molecules and a candidate bank of 277,566 SMILES, the baseline achieved an unacceptable **MRR@25 of 0.0006** and **Recall@25 of 0.50%**.

### 2.2 Mathematical Formulation of Soft Mass Scoring
Mass spectrometry fragment ions represent substructures of a precursor ion whose mass is strictly bounded by the neutral mass of the molecule. Searching across 277,566 unrestricted candidates allowed structurally similar molecules of completely different molecular weights to outscore true isomers.

We introduced **Gaussian Soft Mass Reranking**:
$$\Delta\text{ppm}(c) = \frac{|M_{\text{candidate}}(c) - \bar{M}_{\text{consensus}}|}{\bar{M}_{\text{consensus}}} \times 10^6$$
$$S_{\text{mass}}(c) = \begin{cases} 
\exp\left(-\frac{1}{2}\left(\frac{\Delta\text{ppm}(c)}{\sigma}\right)^2\right) & \text{if } \Delta\text{ppm}(c) \le 50\text{ ppm} \\
0.0 & \text{otherwise}
\end{cases}$$
$$S_{\text{combined}}(s, c) = \langle z_{\text{spec}}^{(s)}, z_{\text{mol}}^{(c)} \rangle \times S_{\text{mass}}(c)$$

### 2.3 Mass-Scoring Ablation Results

We conducted a strictly controlled ablation on identical validation molecules (400 held-out identities, 17,267 spectra, 277,566 candidate bank):

| Scoring Configuration | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Mean Rank | Target Survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Neural Similarity Only** | 0.0005 | 0.00% | 0.00% | 0.25% | 0.50% | 13,842.1 | 100.00% |
| **Mass Compatibility Only ($\sigma = 20\text{ ppm}$)** | 0.0435 | 1.75% | 5.50% | 8.75% | 18.25% | 94.2 | 98.00% |
| **Neural + Mass ($\sigma = 50\text{ ppm}$)** | 0.2520 | 14.50% | 37.00% | 48.25% | 66.50% | 13.8 | 98.00% |
| **Neural + Mass ($\sigma = 30\text{ ppm}$)** | 0.2642 | 15.50% | 39.00% | 49.75% | 68.75% | 13.1 | 98.00% |
| **Neural + Mass ($\sigma = 20\text{ ppm}$)** *(Optimal)* | **0.2686** | **16.00%** | **40.00%** | **50.50%** | **69.75%** | **12.89** | **98.00%** |

**Conclusions:**
1. Neither neural embeddings nor mass compatibility alone can solve molecular identification.
2. When combined multiplicatively, neural embeddings provide fine-grained isomer discrimination among the mass-compatible candidate subset, driving MRR@25 from 0.0006 to 0.2686 (+44,667% relative gain).

---

## 3. Checkpoint Training Duration Benchmark (Steps 1,000 to 20,000)

We evaluated checkpoints saved across the original 20,000-step training trajectory under the optimal mass-aware protocol ($\sigma = 20\text{ ppm}$):

| Checkpoint | Training Steps | Total Presentations | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Val Loss |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| `step_1000.pt` | 1,000 | 128,000 | 0.2412 | 13.75% | 35.50% | 48.00% | 67.25% | 5.124 |
| `step_2000.pt` | 2,000 | 256,000 | 0.2541 | 14.75% | 37.50% | 49.25% | 68.50% | 4.962 |
| `step_5000.pt` *(Champion)* | 5,000 | 640,000 | **0.2686** | **16.00%** | **40.00%** | **50.50%** | **69.75%** | **4.788** |
| `step_10000.pt` | 10,000 | 1,280,000 | 0.2638 | 15.50% | 39.25% | 49.75% | 69.00% | 5.341 |
| `step_15000.pt` | 15,000 | 1,920,000 | 0.2621 | 15.25% | 38.75% | 49.50% | 68.75% | 5.518 |
| `step_20000.pt` | 20,000 | 2,560,000 | 0.2604 | 15.00% | 38.50% | 49.00% | 68.50% | 5.682 |

**Key Finding:** Performance peaked at **Step 5,000** and deteriorated steadily through Step 20,000.

---

## 4. Phase 6A: Root Cause Failure Analysis

An exhaustive audit of the 20,000-step trajectory revealed three core failure mechanisms:

1. **In-Batch Hard-Negative Starvation:**
   In random streaming batches of 128 spectra, 99.86% of in-batch negative pairs had Morgan Tanimoto similarity $T < 0.4$ and wildly discordant precursor masses. The model received essentially zero gradient signal forcing it to discriminate isomers.
2. **Spectral Embedding Dimensional Collapse:**
   Effective rank ($\text{Rank}_{\text{eff}} = \exp(-\sum p_i \ln p_i)$) dropped from 78.1 down to 55.3, condensing representations onto a narrow anisotropic subspace.
3. **Absence of Learning Rate Annealing:**
   Training at a constant $10^{-4}$ learning rate for 20,000 steps without decay caused weights to oscillate around minima, degrading positive validation alignment from $+0.454$ down to $+0.355$.

---

## 5. Phase 6B: Controlled Experiments (Runs A & C)

To address these failure modes, we designed a controlled multi-arm experiment branching strictly from the Step 5,000 champion weights.

### 5.1 Run A: Cosine Warmup Learning-Rate Annealing
- **Hypothesis:** Annealing the learning rate from $10^{-4}$ down to $10^{-6}$ will prevent gradient oscillation and stabilize validation alignment.
- **Result:**
  - Peak MRR@25 = **0.2573** at Step 5,000.
  - Recall@10 = **51.50%** (improved over Champion's 50.50%).
  - Recall@25 = **67.75%**.

### 5.2 Run C: Chemistry-Aware Hard-Negative Mining
- **Hypothesis:** Injecting external hard negatives (~45% mass-compatible within 50 ppm, 27.6% Morgan Tanimoto $T \ge 0.3$) will penalize false isomer matches and improve top-tier ranking.
- **Empirical Negative Distribution (Audited across 1,000 anchors):**
  - Mass-Compatible ($\le 50\text{ ppm}$): 44.81%
  - Morgan Tanimoto ($T \ge 0.3$): 27.59%
  - Random Structural Negatives: 27.59%
  - False Negatives: Exactly 0 (strict identity masking)
  - Silent Substitution: Zero
- **Result:**
  - **Recall@25 Reached an All-Time Project High of 70.50%** at Step 2,000 (beating the Champion's 69.75%).
  - **Recall@10 tied the project high at 51.50%**.
  - **Mean Rank improved to 12.87** (Champion: 12.89).
  - **Target Survival Rate confirmed at 98.00%**.
  - However, **MRR@25 peaked at 0.2525** due to effective rank contraction (63.3 vs 70.6) caused by un-normalized MIL-NCE denominator pressure.

---

## 6. Master Evaluation Benchmark Comparison

| Experiment Arm | Step | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Target Survival | Effective Rank | Validation Loss |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Current Champion (`step_5000.pt`)** | 5,000 | **0.2686** | **16.00%** | **40.00%** | 50.50% | 69.75% | **98.00%** | **70.6** | **4.7882** |
| **Run A (Cosine Baseline)** | 5,000 | 0.2573 | 15.00% | 36.75% | **51.50%** | 67.75% | **98.00%** | 66.1 | 5.2943 |
| **Run C (Step 1,000)** | 1,000 | 0.2396 | 12.75% | 35.50% | 50.25% | 67.00% | **98.00%** | 67.2 | 4.9016 |
| **Run C (Step 2,000)** | 2,000 | **0.2525** | 14.00% | 37.25% | **51.50%** | **70.50%** | **98.00%** | 63.3 | 4.9854 |
| **Run C (Step 3,000)** | 3,000 | 0.2388 | 13.25% | 34.00% | 48.50% | 67.75% | **98.00%** | 63.6 | 5.3086 |
| **Run C (Step 4,000)** | 4,000 | 0.2381 | 13.25% | 35.00% | 49.50% | 64.25% | **98.00%** | 63.9 | 5.4126 |
| **Run C (Step 5,000)** | 5,000 | 0.2418 | 13.50% | 34.75% | 49.25% | 65.25% | **98.00%** | 63.5 | 5.2963 |

---

## 7. Final Champion Selection

Under our pre-registered Decision Protocol (Phase 4):
* Promoting a new checkpoint requires strictly surpassing the benchmark on the competition primary metric (**MRR@25 > 0.2686**).
* While Run C Step 2,000 established the project record for **Recall@25 (70.50%)**, its top-1 accuracy (14.00% vs 16.00%) and MRR@25 (0.2525 vs 0.2686) did not exceed the champion.
* **Selection:** `checkpoints/spectrum_molecule_step_5000.pt` is definitively retained as the official competition champion.

