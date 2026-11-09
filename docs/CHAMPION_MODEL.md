# Enveda CASMI 2026: Official Champion Model Specification

**Document ID:** `docs/CHAMPION_MODEL.md`  
**Official Champion Checkpoint:** `checkpoints/spectrum_molecule_step_5000.pt`  
**Status:** Pre-registered, verified, and protected project champion.  

---

## 1. Executive Summary & Selection Rationale

The official champion model for the Enveda CASMI 2026 competition is:
`checkpoints/spectrum_molecule_step_5000.pt`

### Official Validated Performance
Evaluated across the full **277,566 candidate SMILES bank** over all **400 held-out validation molecules** (17,267 MS/MS spectra) using **Mass-Aware Mean Scoring ($\sigma = 20\text{ ppm}$)**:

* **MRR@25:** **0.2686** (Benchmark Leader)
* **Recall@1:** **16.00%** (Highest top-1 precision)
* **Recall@5:** **40.00%**
* **Recall@10:** **50.50%**
* **Recall@25:** **69.75%**
* **Mean Rank:** **12.89**
* **Median Rank:** **10.0**
* **Effective Rank ($\text{Rank}_{\text{eff}}$):** **70.57** (Optimal embedding representation isotropy)
* **Validation Loss:** **4.7882** (Lowest validation loss across all evaluated checkpoints)

---

## 2. Why This Model Was Selected & Rejection of Alternatives

Under our pre-registered experimental evaluation protocol, a candidate model could only supplant the champion if it strictly exceeded **MRR@25 > 0.2686** on identical validation conditions.

### 2.1 Rejection of Extended Training Checkpoints (Steps 10,000 to 20,000)
* **Step 10,000:** MRR@25 dropped to **0.2638**, validation loss rose to **5.341**, and effective rank contracted to **64.2**.
* **Step 20,000:** MRR@25 fell further to **0.2604**, validation loss deteriorated to **5.682**, and effective rank collapsed to **55.3**.
* **Diagnosis:** Extended training at a constant learning rate ($10^{-4}$) without decay caused positive alignment degradation ($+0.454 \to +0.355$) and in-batch hard negative starvation.

### 2.2 Rejection of Phase 6B Run A (Cosine Warmup Baseline)
* **Best Checkpoint (Step 5,000):** MRR@25 peaked at **0.2573**, below the champion's **0.2686**.
* **Recall Metrics:** Recall@1 was 15.00% (vs. 16.00% for the champion), and Recall@25 was 67.75% (vs. 69.75% for the champion).

### 2.3 Rejection of Phase 6B Run C (Chemistry-Aware Negative Training)
* **Best Checkpoint (Step 2,000):** Achieved an all-time project record for **Recall@25 of 70.50%** (outperforming the champion's 69.75%) and tied Recall@10 at **51.50%**.
* **Failure on Primary Metric:** However, its MRR@25 peaked at **0.2525** and top-1 accuracy was **14.00%** (versus the champion's **16.00%** and **0.2686**).
* **Root Cause:** In standard un-normalized MIL-NCE, multi-positive anchors experienced an unbalanced denominator pull when hard negatives were injected, compressing effective rank from 70.6 to 63.3. Under Decision Rule Case C, the champion was strictly retained.

---

## 3. Detailed Model Architecture & Provenance

### 3.1 Spectrum Encoder (`SpectrumEncoder`)
* **Input Features:** 10,130 dimensions (10,000 sparse binned peaks at 0.1 Da resolution + 130 tabular experimental metadata features).
* **Architecture:** 3-layer MLP with LayerNorm, GELU activations, and Dropout:
  $$\text{Linear}(10130 \to 512) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.2)$$
  $$\to \text{Linear}(512 \to 256) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.1)$$
  $$\to \text{Linear}(256 \to 128) \to \text{L2 Normalization}$$
* **Output:** 128-dimensional unit vector $z_{\text{spec}} \in \mathbb{S}^{127}$.

### 3.2 Molecule Encoder (`MoleculeEncoder`)
* **Input Features:** 2048-bit Morgan circular fingerprints (ECFP4, radius 2, binary float vector).
* **Architecture:** 3-layer MLP:
  $$\text{Linear}(2048 \to 512) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.2)$$
  $$\to \text{Linear}(512 \to 256) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.1)$$
  $$\to \text{Linear}(256 \to 128) \to \text{L2 Normalization}$$
* **Output:** 128-dimensional unit vector $z_{\text{mol}} \in \mathbb{S}^{127}$.

### 3.3 Training Hyperparameters & Provenance
* **Total Presentation Budget:** 640,000 spectrum presentations (5,000 steps $\times$ batch size 128).
* **Loss Function:** Multi-positive symmetric contrastive loss (MIL-NCE) with temperature $\tau = 0.07$.
* **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay = $10^{-4}$, initial learning rate = $10^{-4}$).
* **Mixed Precision:** PyTorch AMP (`GradScaler`).
* **Checkpoint File Size:** 78,881,515 bytes (~75.2 MB).

---

## 4. Preprocessing & Retrieval Pipeline

### 4.1 Spectrum Preprocessing
1. Relative intensity thresholding: peak intensity $I / I_{\max} \ge 0.01$.
2. Top-128 peak selection: discard low-intensity noise past peak rank 128.
3. Discretization: 10,000 uniform bins ($\Delta m/z = 0.1\text{ Da}$) spanning $[0.0, 1000.0\text{ Da}]$.
4. Vector L2 normalization: $\|x_{\text{spec}}\|_2 = 1.0$.

### 4.2 Candidate Bank Specification
* **Total Candidates:** 277,566 unique molecular structures represented as canonical SMILES.
* **Precomputed Artifacts:**
  * Embeddings: `outputs/candidate_bank/candidate_embeddings.npy` (Shape: `[277566, 128]`, `float32`).
  * Neutral Masses: `outputs/candidate_bank/candidate_exact_masses.npy` (Shape: `[277566]`, `float32`).
  * Manifest: `outputs/candidate_bank/candidate_manifest.json` (List of 277,566 SMILES).

### 4.3 Mass-Aware Mean Scoring
For each query molecule $m$ with spectra $S_m$ and candidate $c$:
1. Calculate **Consensus Neutral Mass** across query spectra:
   $$\bar{M}_{\text{consensus}} = \text{median}\left(\left\{\frac{m/z_{\text{prec}}^{(s)} \cdot |z_s| - \Delta m_{\text{adduct}}^{(s)}}{1.0}\right\}_{s \in S_m}\right)$$
2. Compute relative error $\Delta\text{ppm}(c) = \frac{|M_{\text{candidate}}(c) - \bar{M}_{\text{consensus}}|}{\bar{M}_{\text{consensus}}} \times 10^6$.
3. Compute Gaussian soft mass compatibility multiplier:
   $$S_{\text{mass}}(c) = \begin{cases} \exp\left(-\frac{1}{2}\left(\frac{\Delta\text{ppm}(c)}{20.0}\right)^2\right) & \text{if } \Delta\text{ppm}(c) \le 50\text{ ppm} \\ 0.0 & \text{otherwise} \end{cases}$$
4. Aggregate mass-aware score across query spectra:
   $$S_{\text{agg}}(m, c) = \frac{1}{|S_m|} \sum_{s \in S_m} \langle z_{\text{spec}}^{(s)}, z_{\text{mol}}^{(c)} \rangle \times S_{\text{mass}}(c)$$
5. Rank candidates in descending order of $S_{\text{agg}}(m, c)$ and output top 25 SMILES.

---

## 5. Investigation of Metric Discrepancies: Target Survival vs Recall@25

### 5.1 The Discrepancy
In early progress notes and initial summary drafts, the figure **69.75%** was occasionally conflated with "target survival", while later reports cited **98.00%** target survival.

### 5.2 Implementation Audit & True Definitions
An inspection of `src/enveda/evaluation/metrics.py`, `src/enveda/evaluation/mass_filter.py`, and `outputs/checkpoint_comparison.json` reveals the exact cause of this ambiguity:

1. **Target Survival Rate (98.00%):**
   * *Formula:* $\frac{1}{M} \sum_{m=1}^M \mathbb{I}(\Delta\text{ppm}(c_m^*) \le 50\text{ ppm})$
   * *Measured Value:* Exactly 392 of 400 validation targets have neutral mass within 50 ppm of the consensus precursor mass ($\mathbf{98.00\%}$). The remaining 8 targets ($2.00\%$) deviate by $> 50\text{ ppm}$ due to uncataloged adducts or extreme instrument mass deltas.
   * *Meaning:* Target survival measures the *theoretical recall ceiling* imposed by candidate screening.
2. **Recall@25 (69.75%):**
   * *Formula:* $\frac{1}{M} \sum_{m=1}^M \mathbb{I}(\text{rank}(c_m^*) \le 25)$
   * *Measured Value:* Exactly 279 of 400 validation targets were ranked in the top 25 candidates ($\mathbf{69.75\%}$).
   * *Meaning:* Recall@25 measures the *actual retrieval performance* of the neural model and reranker.
3. **Unconstrained Soft Scoring (100.0% Survival):**
   * In `docs/RETRIEVAL_IMPROVEMENT_REPORT.md`, when the Gaussian multiplier was evaluated across all 277,566 candidates *without hard pruning*, zero candidates were discarded, yielding **100.0% survival**, but with the identical **Recall@25 of 69.75%**.

**Conclusion:** 69.75% was never target survival; it was the model's **Recall@25**. The true candidate survival within the 50 ppm screening window is strictly **98.00%**. All documentation has been aligned to these verified definitions.
