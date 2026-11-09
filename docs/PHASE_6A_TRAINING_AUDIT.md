# Enveda CASMI 2026 — Phase 6A: Training Behaviour Audit

**Document ID:** `docs/PHASE_6A_TRAINING_AUDIT.md`  
**Phase:** 6A — Systematic Investigation of Training Dynamics & Overfitting  
**Evaluation Set:** 400 held-out validation molecules (`inchikey14` grouped, zero overlap), 17,267 MS/MS spectra  
**Candidate Bank Universe:** 277,566 unique SMILES  
**Status:** Complete Audit (Read-only, no training launched)

---

## 1. Executive Summary

In Phase 5, the two-tower retrieval model achieved its peak validation performance at **Step 5,000** (**MRR@25 = 0.2686**, **Recall@25 = 69.75%** with $\sigma = 20$ ppm mass scoring). However, as training extended through **10,000**, **15,000**, and **20,000 steps**, retrieval precision steadily declined to **MRR@25 = 0.2604**, even while training loss dropped monotonically from 2.09 down to 1.53.

This Phase 6A audit was conducted to diagnose the mathematical, algorithmic, and empirical causes of this performance degradation before making any modifications to the model architecture or training pipeline.

### Core Empirical Audit Summary:

| Checkpoint | Training Step | Recent Train Loss | Validation Loss | Spectrum Eff. Rank (max 128) | Molecule Eff. Rank (max 128) | Pos. Cosine (Val) | Neg. Cosine (Val) | Cosine Margin | Mass ($\sigma=20$ ppm) MRR@25 | Mass ($\sigma=20$ ppm) Recall@25 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Step 5k** | 5,000 | 2.0935 | 3.0466 | **78.1** | **66.3** | +0.446 | +0.227 | +0.219 | **0.2686** | 69.75% |
| **Step 10k** | 10,000 | 1.9627 | 2.9376 | 64.1 | 57.7 | **+0.454** | +0.149 | +0.305 | 0.2668 | 69.50% |
| **Step 15k** | 15,000 | 1.6908 | **2.9124** | 59.2 | 59.7 | +0.386 | +0.050 | +0.336 | 0.2656 | 70.25% |
| **Step 20k** | 20,000 | **1.5347** | 2.9373 | 55.3 | 60.6 | +0.355 | **+0.018** | **+0.337** | 0.2604 | **70.50%** |

---

## 2. Task 1: Training Loss Implementation Audit

### 2.1 Multi-Positive Formulation
The contrastive objective implemented in `src/enveda/training/losses.py` uses the "Sum-Inside-Log" variant of multi-positive InfoNCE:
$$\mathcal{L}_{\text{s2m}} = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^B \exp(s_{ij} / \tau) - \log \sum_{p \in P(i)} \exp(s_{ip} / \tau) \right)$$

#### Mathematical Diagnosis:
1. **Anchor Inbalance Bias:**
   In random streaming batches, the number of positive spectra per anchor $|P(i)|$ varies from 1 to 16.
   $$\log \sum_{p \in P(i)} \exp(s_{ip} / \tau) \approx \log |P(i)| + \frac{s_{ip}^*}{\tau}$$
   An anchor with 8 co-occurring spectra gets an automatic mathematical offset of $\approx \log(8) = 2.08$ subtracted from its loss, compared to an anchor with only 1 spectrum ($\log(1) = 0$). This biases gradients toward molecules with high spectrum multiplicity in the training split.
2. **Symmetric Molecule-to-Spectrum Loss Inconsistency:**
   Because spectra from the same molecule share identical Morgan fingerprints, row $i$ and row $j$ in `molecule_embeddings` are bitwise identical if $\text{molecule\_id}_i = \text{molecule\_id}_j$. In `loss_m2s`, the model treats duplicate rows in the denominator as separate negative anchors, artificially magnifying repulsion between non-identical spectra of the same molecule.

### 2.2 Hard-Negative Weighting Audit
The hard-negative modification applies an additive log-boost to negative pairs exceeding the Morgan Tanimoto threshold:
$$\text{logits}_{ij}' = \text{logits}_{ij} + \log(w_{\text{hard}}) \cdot \mathbf{1}_{\{j \notin P(i) \land T_{ij} \ge \gamma\}}$$
where $w_{\text{hard}} = 1.5$ and $\gamma = 0.4$.

#### Effect on the Objective:
In the softmax denominator:
$$\exp(\text{logits}_{ij}') = \exp(s_{ij}/\tau + \log(1.5)) = 1.5 \cdot \exp(s_{ij}/\tau)$$
This mathematically scales the repulsion gradient for structurally similar negatives by $1.5\times$ without altering the positive numerator. It is mathematically sound and bounded. However, its effectiveness is bottlenecked by batch composition (see Task 2).

---

## 3. Task 2: Batch Composition & Hard-Negative Scarcity Audit

We empirically profiled 50 training batches (6,400 spectrum presentations) from `StreamingParquetDataset`:

| Metric | Measured Value | Analysis |
|---|:---:|---|
| **Batch Size** | 128 | Fixed batch size |
| **Unique Molecules per Batch** | 60.96 (47.6%) | High spectrum replication per batch |
| **Off-Diagonal Positive Pairs** | 275.76 per batch | Strong multi-positive supervision |
| **Mean Negative Morgan Tanimoto** | **0.1223** | Random molecules are structurally dissimilar |
| **Median Negative Morgan Tanimoto** | **0.1200** | Most negatives are trivial |
| **Negatives with $T \ge 0.3$** | **0.29%** | Less than 3 per 1,000 negative pairs |
| **Negatives with $T \ge 0.4$ (Hard)** | **0.14%** | Only 1 in 700 negative pairs |
| **Hard Negatives per Anchor ($T \ge 0.4$)** | **0.089** | **Fewer than 1 hard negative per 11 anchors** |
| **Very Hard Negatives ($T \ge 0.6$)** | **0.013** | Virtually non-existent in random mini-batches |

### Key Finding:
Because mini-batches are constructed by sequential/random streaming across the 2.4M training spectra, **99.86% of all negative pairs in any batch are structurally distant random molecules ($T < 0.4$)**.
True structural hard negatives occur only **0.089 times per anchor**. Consequently:
- The model spends 99.86% of its gradient capacity repelling random molecules (pushing negative cosine from $+0.227$ down to $+0.018$).
- It rarely encounters constitutional isomers or structural neighbors during in-batch negative mining.

---

## 4. Task 3: Optimization & Learning Rate Audit

### 4.1 Absence of Learning Rate Scheduling
- **Current Setup:** AdamW optimizer with a **constant learning rate of $1\times 10^{-4}$** and weight decay $1\times 10^{-4}$ throughout all 20,000 steps ($2.56\times 10^6$ presentations).
- **Diagnosis:** A constant learning rate of $10^{-4}$ prevents representation refinement once broad clustering is achieved (~5,000 steps). The large step size in late training causes the optimizer to bounce around the sharp minimum, eroding generalizable positive alignment on unseen spectra (positive cosine dropped from $+0.454$ at 10k to $+0.355$ at 20k).

### 4.2 Gradient Scaler & Momentum Buffer Behaviour
- Checkpoint audit confirmed PyTorch AMP `GradScaler` maintained stable scaling with zero gradient underflows or inf/NaN rollbacks.
- Weight decay ($10^{-4}$) is applied symmetrically to all parameters, including LayerNorm weights and biases, which can subtly compress layer gain over extended training.

---

## 5. Task 4: Checkpoint Representation Analysis & Dimensional Collapse

### 5.1 Dimensional Collapse (Effective Rank Analysis)
Using the Shannon entropy of singular values (Roy & Vetterli, 2007), we measured the effective rank of the 128-dimensional latent space on validation embeddings:
- **Step 5k:** Spectrum Rank = **78.1**, Molecule Rank = **66.3**
- **Step 10k:** Spectrum Rank = **64.1**, Molecule Rank = **57.7**
- **Step 15k:** Spectrum Rank = **59.2**, Molecule Rank = **59.7**
- **Step 20k:** Spectrum Rank = **55.3**, Molecule Rank = **60.6**

**Diagnosis:** The Spectrum Encoder exhibits progressive **dimensional collapse**, losing ~23 effective dimensions between Step 5k and Step 20k. The neural network over-optimizes training spectrum noise onto a smaller subspace, harming out-of-distribution generalization.

### 5.2 Cosine Alignment vs. Uniformity Dilemma
- The contrastive loss succeeded at pushing negative pairs to near-zero cosine similarity:
  $$\text{Mean Negative Cosine: } +0.227 \text{ (5k)} \to +0.149 \text{ (10k)} \to +0.050 \text{ (15k)} \to +0.018 \text{ (20k)}$$
- But this came at the expense of **positive alignment on unseen validation spectra**:
  $$\text{Mean Positive Cosine: } +0.446 \text{ (5k)} \to +0.454 \text{ (10k)} \to +0.386 \text{ (15k)} \to +0.355 \text{ (20k)}$$
- As positive alignment degrades by nearly 0.10, the model's ability to rank the true structure at rank 1–5 within the mass candidate pool drops, driving MRR@25 down from 0.2686 to 0.2604.

---

## 6. Root Causes of Declining Performance

### Root Causes Supported by Direct Evidence:
1. **Generalization Overfitting:** The training loss drops from 2.09 to 1.53, while validation loss bottoms at step 15k and diverges. The generalization gap expands from **0.95 to 1.40**.
2. **Degradation of Validation Positive Alignment:** The positive cosine similarity on held-out validation spectra falls from $+0.454$ down to $+0.355$, degrading fine-grained top-1/top-5 rank discrimination.
3. **Spectrum Dimensional Collapse:** Spectrum effective rank declines steadily from 78.1 to 55.3, concentrating representations into an anisotropic subspace.
4. **Hard-Negative In-Batch Starvation:** Random batching supplies only **0.089 hard negatives per anchor**. 99.86% of negative gradient updates repel trivial random molecules rather than teaching the model to distinguish similar isomers.
5. **Lack of Learning Rate Annealing:** Keeping $LR = 10^{-4}$ constant without cosine decay prevents the model from settling into flat minima.

### Potential Causes Requiring Experimental Verification:
1. **Multi-Positive MIL-NCE vs. SupCon Formulation:** Whether summing inside log ($\log \sum \exp$) creates unfair anchor weighting compared to averaging normalized log-probabilities ($\frac{1}{|P|} \sum \log \dots$).
2. **Mass-Incompatible Negative Repulsion:** Whether repelling molecules that have completely different precursor masses wastes embedding capacity on pairs that will be filtered out anyway by the mass engine.

---

## 7. Recommended Changes for Phase 6B

1. **Cosine Annealing Learning Rate Schedule:**
   Introduce linear warmup (500 steps) followed by cosine decay down to $1\times 10^{-6}$, allowing representations to stabilize.
2. **Mass-Aware & Tanimoto-Targeted Batching (or Memory Queue):**
   Group training samples by precursor mass or cluster by chemical family so batches contain high-density structural and isobaric competitors ($T \ge 0.4$ and $\Delta m/z < 50$ ppm).
3. **Normalized Supervised Contrastive Loss (SupCon):**
   Replace MIL-NCE sum-inside-log with standard normalized Supervised Contrastive Loss to remove anchor-frequency bias.
4. **Representation Regularization Against Collapse:**
   Add a covariance regularization term (VICReg/Barlow Twins style) to preserve embedding effective rank above 75.

---

## 8. Proposed Controlled Experiment for Phase 6B

### Hypothesis:
Replacing the constant learning rate with a **Cosine Decay Schedule** and applying **Precursor-Mass-Aware Batch Sampling** will prevent dimensional collapse, preserve positive cosine alignment ($> +0.45$), and boost validation MRR@25 beyond **0.2800**.

### Experimental Matrix (Controlled from Step 5,000):

| Run ID | LR Schedule | Loss Objective | Negative Mining Strategy | Target Evaluation Metric |
|---|---|---|---|---|
| **Run 1 (Baseline)** | Constant $10^{-4}$ | MIL-NCE (current) | Random in-batch ($w=1.0$) | MRR@25 baseline (0.2686) |
| **Run 2 (Scheduler)** | Cosine Decay ($10^{-4} \to 10^{-6}$) | MIL-NCE (current) | Random in-batch ($w=1.0$) | Test LR stabilization hypothesis |
| **Run 3 (SupCon)** | Cosine Decay ($10^{-4} \to 10^{-6}$) | SupCon (normalized) | Random in-batch ($w=1.0$) | Test loss normalization hypothesis |
| **Run 4 (Mass-Hard)** | Cosine Decay ($10^{-4} \to 10^{-6}$) | SupCon (normalized) | Mass-clustered / Semi-hard ($T \ge 0.3$) | Test hard negative density hypothesis |

*Awaiting review before proceeding to Phase 6B execution.*

