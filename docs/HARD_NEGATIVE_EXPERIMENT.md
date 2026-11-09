# Enveda CASMI 2026 — Phase 5C: Hard Negative Mining Investigation

**Date:** November 8, 2026  
**Auditor:** Senior ML Research Engineer  
**Scope:** In-batch hard negative mining using GPU-accelerated Morgan Tanimoto similarity for contrastive retrieval.

---

## 1. Motivation & Problem Formulation

In standard contrastive learning (`multi_positive_contrastive_loss`), negative pairs are formed from randomly sampled in-batch spectra:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\sum_{j \in P(i)} \exp(\text{sim}(s_i, m_j) / \tau)}{\sum_{j \in P(i)} \exp(\text{sim}(s_i, m_j) / \tau) + \sum_{k \notin P(i)} \exp(\text{sim}(s_i, m_k) / \tau)}$$

### The Random Negatives Limitation:
- When drawing random batches of size 128 from 275,810 unique molecules, the vast majority of negative pairs are chemically completely unrelated (e.g., Tanimoto similarity $< 0.10$).
- The model quickly achieves low contrastive loss by easily separating unrelated chemotypes (e.g. lipids from small aromatics).
- However, during retrieval over 277,566 candidates, the most difficult errors arise from **structural analogs, positional isomers, and near-homologs** (Tanimoto similarity $\ge 0.40$ with different molecular skeletons).
- Without hard negatives, the embedding space does not learn sufficient fine-grained resolution to distinguish closely related candidate structures.

---

## 2. Hard Negative Mining Architecture

### 2.1 GPU-Accelerated Batch Morgan Tanimoto Calculation
Given a batch of 2,048-bit Morgan fingerprints $F \in \{0, 1\}^{B \times 2048}$:
$$\text{Intersection} = F F^T \in \mathbb{R}^{B \times B}$$
$$\text{Union}_{ij} = \|F_i\|_1 + \|F_j\|_1 - \text{Intersection}_{ij}$$
$$\text{Tanimoto}_{ij} = \frac{\text{Intersection}_{ij}}{\text{Union}_{ij}}$$
- **Computational Efficiency:** Implemented entirely in PyTorch using tensor operations on GPU (`src/enveda/training/losses.py`).
- **Speed:** Less than 0.5 ms per batch of 128 fingerprints on NVIDIA RTX 3050 GPU.
- **Accuracy:** Numerically verified to match RDKit's C++ `DataStructs.TanimotoSimilarity` to $10^{-5}$ tolerance.

### 2.2 Safety Condition: Zero False Negatives
A critical pitfall in hard negative mining is treating matching molecular identities as negatives.
- **Rule:** Positive identities are defined strictly by matching `molecule_ids` (`inchikey14` skeleton hash).
  $$\text{positive\_mask}_{ij} = (\text{molecule\_ids}_i == \text{molecule\_ids}_j)$$
  $$\text{hard\_neg\_mask}_{ij} = (\neg \text{positive\_mask}_{ij}) \land (\text{Tanimoto}_{ij} \ge \tau_{\text{hard}})$$
- **Mathematical Guarantee:** $\text{positive\_mask}_{ij} \cap \text{hard\_neg\_mask}_{ij} = \emptyset$.
- Even if two molecules share $100\%$ structural features, if their `molecule_ids` match, they are **never** treated as negatives.

### 2.3 Logit-Space Gradient Upweighting
To upweight hard negatives smoothly without numerical instability:
$$\text{logit\_boost}_{ij} = \begin{cases} \log(w_{\text{hard}}) & \text{if } \text{hard\_neg\_mask}_{ij} = \text{True} \\ 0 & \text{otherwise} \end{cases}$$
$$\text{logits}_{ij}^{\text{mod}} = \frac{\text{sim}(s_i, m_j)}{\tau} + \text{logit\_boost}_{ij}$$
- For $w_{\text{hard}} = 1.0$: $\log(1.0) = 0$, identically recovering the baseline contrastive loss.
- For $w_{\text{hard}} = 2.0$: hard negatives exert $2\times$ the repulsive probability mass in the denominator, concentrating gradients on hard structural distractors.

---

## 3. Integration & Unit Test Verification

- Integrated into:
  - `src/enveda/training/losses.py`: `compute_batch_tanimoto`, `multi_positive_contrastive_loss`.
  - `src/enveda/training/trainer.py`: automatic batch fingerprint extraction and loss routing via `config.training.hard_negative_weight` and `config.training.hard_tanimoto_threshold`.
- Tested in `tests/test_model.py::test_hard_negative_mining`.
- Verified 25/25 unit tests passing.

