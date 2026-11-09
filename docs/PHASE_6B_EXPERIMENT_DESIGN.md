# Enveda CASMI 2026 — Phase 6B: Chemistry-Aware Training & Controlled Experiment Design

**Document ID:** `docs/PHASE_6B_EXPERIMENT_DESIGN.md`  
**Phase:** 6B — Chemistry-Aware Contrastive Learning & Controlled Experiment Design  
**Current Champion:** `checkpoints/spectrum_molecule_step_5000.pt`  
**Champion Benchmark:** MRR@25 = **0.2686**, Recall@25 = **69.75%** (Mass-Aware Mean, $\sigma = 20$ ppm)  
**Evaluation Set:** 400 held-out validation molecules (`inchikey14` grouped, zero overlap), 17,267 MS/MS spectra  
**Candidate Bank Universe:** 277,566 unique SMILES  
**Status:** All Audits, Empirical Validations & Smoke Tests Verified — Awaiting Approval Before Training Launch

---

## 1. Executive Summary & Context

Phase 6A diagnosed three core failure modes that caused validation retrieval performance to degrade from Step 5,000 to Step 20,000 (MRR@25 falling from 0.2686 to 0.2604):
1. **In-Batch Hard-Negative Starvation:** 99.86% of negatives in random streaming batches are structurally distant random molecules ($T < 0.4$), with only 0.089 hard negatives per anchor. The model wasted 99.86% of its gradient updates pushing trivial negatives towards zero similarity rather than learning fine-grained isomer discrimination.
2. **Spectrum Embedding Dimensional Collapse:** The Spectrum Encoder's effective rank dropped steadily from 78.1 down to 55.3, condensing representations onto an anisotropic subspace.
3. **Validation Positive Alignment Degradation:** Positive cosine similarity between held-out spectra and true molecules deteriorated from $+0.454$ at step 10k to $+0.355$ at step 20k due to the absence of learning rate annealing (constant LR of $10^{-4}$ for 20,000 steps).

**Objective of Phase 6B:**  
Design, implement, mathematically verify, and unit-test the modular components for a controlled 4-arm experiment starting from the Step 5,000 checkpoint, without altering the champion checkpoint.

---

## 2. Component Implementations & Audited Corrections

### 2.1 Experiment 1: Learning-Rate Scheduling (`src/enveda/training/schedulers.py`)
- **Architecture:** `CosineWarmupScheduler` inheriting from PyTorch `_LRScheduler`.
- **Warmup Schedule:** 500-step linear warmup from $\text{min\_lr} = 1\times 10^{-6}$ to $\text{base\_lr} = 1\times 10^{-4}$.
- **Decay Schedule:** Cosine annealing decay from $1\times 10^{-4}$ to $1\times 10^{-6}$ over the training cycle:
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - W}{T - W}\pi\right)\right)$$
- **Audit Diagnosis & Fix (GradScaler Dynamics):**
  - *Diagnosis:* PyTorch's default `init_scale=65536.0` caused initial loss ($65536 \times 5.0 = 327680$) to exceed FP16 maximum dynamic range ($65504$), resulting in inf gradients that halved the scale and skipped optimizer/scheduler steps during steps 0 and 1.
  - *Correction:* Initialized `GradScaler` with `init_scale=8192.0`. Steps 0 through 10 execute without skipping, advancing learning rates strictly from step 0.
- **Milestone Empirical Verification:**
  - Step 0: $\text{lr} = 1.000\times 10^{-6}$
  - Step 1: $\text{lr} = 1.198\times 10^{-6}$
  - Step 2: $\text{lr} = 1.396\times 10^{-6}$
  - Step 10: $\text{lr} = 2.980\times 10^{-6}$
  - Step 250: $\text{lr} = 5.050\times 10^{-5}$
  - Step 500: $\text{lr} = 1.000\times 10^{-4}$ (peak)
  - Step 2,750: $\text{lr} = 5.050\times 10^{-5}$ (cosine midpoint)
  - Step 5,000: $\text{lr} = 1.000\times 10^{-6}$ (decay floor)

### 2.2 Experiment 2: Chemistry-Aware Hard-Negative Sampler (`src/enveda/data/negative_sampler.py`)
- **Module:** `ChemistryAwareNegativePool`.
- **Audit Diagnosis & Fix (Root Cause of Missing Negatives):**
  - An audit over 1,000 representative training anchors revealed that the 57 missing negative slots in a 128-batch presentation (199 sampled vs 256 requested) are **100% attributable to mass-category shortfalls**:
    - **Morgan Tanimoto Structural:** 500 / 500 fulfilled (**0 shortfalls, 100.0%**)
    - **Random Negatives:** 500 / 500 fulfilled (**0 shortfalls, 100.0%**)
    - **Mass-Compatible:** 812 / 1,000 fulfilled (**188 shortfalls in 50k pool, 81.2%**)
    - **Duplicate Removals:** 0
    - **Identity Exclusions:** 206 instances correctly masked
  - *Sampling Policy:* In strict adherence to policy, when an anchor lacks a mass-compatible candidate within 50 ppm, the system **never silently substitutes random noise**. Instead, it returns the available candidate(s).
  - *Actual Measured Distribution:*
    - Mass-Compatible: **44.81%**
    - Morgan Tanimoto Structural: **27.59%**
    - Random: **27.59%**
  - *Per-Anchor Fulfillment:*
    - Exactly 2 negatives: **81.2% of anchors**
    - Exactly 1 negative: **18.8% of anchors**
    - 0 negatives: **0.0% of anchors**
    - Mean negatives per anchor: **1.812**
- **Empirical Morgan Tanimoto Distribution (500 Training Anchors vs 9,890 Candidate Pool):**
  - Max Tanimoto to nearest neighbor: Min = 0.2353, 25% = 0.3974, Median = 0.4627, Mean = 0.4797, 75% = 0.5486, Max = 0.9412
  - $T \ge 0.2$: 100.0% of anchors have matches (mean = 541.8, median = 449.0)
  - $T \ge 0.3$: 95.6% of anchors have matches (mean = 20.3, median = 15.0)
  - $T \ge 0.4$: 74.6% of anchors have matches (mean = 2.9, median = 2.0)
  - $T \ge 0.5$: 38.6% of anchors have matches (mean = 0.8, median = 0.0)
  - *Adaptive Fallback:* If an anchor belongs to the rare 4.4% of scaffolds with no candidates at $T \ge 0.3$, the sampler selects the nearest structural neighbor in the pool ($T \in [0.235, 0.300]$) rather than substituting random noise.

### 2.3 Experiment 3: Normalized Supervised Contrastive Objective (`src/enveda/training/losses.py`)
- **Module:** `normalized_supervised_contrastive_loss`.
- **Formulation (Khosla et al., 2020):**
  $$\mathcal{L}_{\text{s2m}}(i) = \log\left(\sum_{j=1}^B \exp(z_{ij}')\right) - \frac{1}{|P(i)|} \sum_{p \in P(i)} z_{ip}$$
  $$\mathcal{L}_{\text{SupCon}} = \frac{1}{2} \left( \frac{1}{B} \sum_{i=1}^B \mathcal{L}_{\text{s2m}}(i) + \frac{1}{B} \sum_{k=1}^B \mathcal{L}_{\text{m2s}}(k) \right)$$
- **Verified Advantages:** Normalizing by $1/|P(i)|$ prevents anchor multiplicity bias where molecules with 8–16 spectra dominate batch gradients.

### 2.4 Experiment 4: Embedding Diagnostics Tracker (`src/enveda/training/diagnostics.py`)
- Tracks spectrum and molecule effective rank ($\text{Rank}_{\text{eff}}$), feature variance, cosine similarities, and validation generalization gap.

---

## 3. Controlled Experiment Design (Phase 6B Protocol)

### 3.1 Four Experimental Configurations

| Run ID | Name | Learning Rate Schedule | Contrastive Loss Formulation | Negative Sampling Strategy | Target Hypothesis |
|:---:|---|---|---|---|---|
| **Run A** | *Cosine Baseline* | Cosine Warmup ($10^{-4} \to 10^{-6}$) | Baseline MIL-NCE (`multi_positive`) | Standard in-batch random | Isolates the impact of LR annealing on positive alignment |
| **Run B** | *Normalized SupCon* | Cosine Warmup ($10^{-4} \to 10^{-6}$) | Normalized SupCon (`normalized_supcon`) | Standard in-batch random | Tests whether $1/|P|$ normalization prevents multiplicity overfitting |
| **Run C** | *Chemistry Negatives* | Cosine Warmup ($10^{-4} \to 10^{-6}$) | Baseline MIL-NCE (`multi_positive`) | Chemistry-Aware Pool (~45% Mass, 27.5% Tanimoto, 27.5% Rand) | Tests whether mass-compatible hard negatives sharpen isomer ranking |
| **Run D** | *Full Chemistry-Aware* | Cosine Warmup ($10^{-4} \to 10^{-6}$) | Normalized SupCon (`normalized_supcon`) | Chemistry-Aware Pool (~45% Mass, 27.5% Tanimoto, 27.5% Rand) | Tests synergistic combination of normalized loss + chemical negatives |

### 3.2 Controlled Execution Constraints
- **Common Origin:** All four runs resume strictly from the verified champion checkpoint: `checkpoints/spectrum_molecule_step_5000.pt`.
- **Fresh Local Step Counter:** Each run uses a fresh 5,000-step counter (Step 0 to Step 5,000 local steps, matching Steps 5,000 to 10,000 cumulative).
- **Independent Optimization State:** Each run initializes its own clean optimizer, scaler (`init_scale=8192.0`), and scheduler state dicts to prevent cross-run state contamination.
- **Identical Training Budget:** Exactly 5,000 steps per run, corresponding to 640,000 spectrum presentations (batch size 128).
- **Identical Train / Val Splits:** Standard `data/cache/group_split.json` (zero identity leakage).
- **Isolated Checkpoint Directories:**
  - Run A: `checkpoints/phase6b_run_a/`
  - Run B: `checkpoints/phase6b_run_b/`
  - Run C: `checkpoints/phase6b_run_c/`
  - Run D: `checkpoints/phase6b_run_d/`
  - Original `checkpoints/spectrum_molecule_step_5000.pt` remains untouched.
- **Evaluation Benchmark:** Evaluated on the held-out 400 validation molecules using Mass-Aware Mean scoring at $\sigma = 20$ ppm.

---

## 4. Controlled 10-Step Optimizer Update Verification

A controlled 10-step training verification was executed on GPU with `init_scale=8192.0`:

| Step | Loss | Grads Finite | Scale Factor | Opt Skipped | Weight Delta ($\| \Delta \theta \|_2$) | LR Progression |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 5.0307 | True | 8192 $\to$ 8192 | **False** | $8.156\times 10^{-5}$ | $0.00000100 \to 0.00000120$ |
| **2** | 4.5405 | True | 8192 $\to$ 8192 | **False** | $8.786\times 10^{-5}$ | $0.00000120 \to 0.00000140$ |
| **3** | 4.5723 | True | 8192 $\to$ 8192 | **False** | $9.665\times 10^{-5}$ | $0.00000140 \to 0.00000159$ |
| **4** | 4.7663 | True | 8192 $\to$ 8192 | **False** | $9.877\times 10^{-5}$ | $0.00000159 \to 0.00000179$ |
| **5** | 4.8532 | True | 8192 $\to$ 8192 | **False** | $1.103\times 10^{-4}$ | $0.00000179 \to 0.00000199$ |
| **6** | 4.7804 | True | 8192 $\to$ 8192 | **False** | $1.252\times 10^{-4}$ | $0.00000199 \to 0.00000219$ |
| **7** | 4.7151 | True | 8192 $\to$ 8192 | **False** | $1.371\times 10^{-4}$ | $0.00000219 \to 0.00000239$ |
| **8** | 4.8916 | True | 8192 $\to$ 8192 | **False** | $1.478\times 10^{-4}$ | $0.00000239 \to 0.00000258$ |
| **9** | 5.2227 | True | 8192 $\to$ 8192 | **False** | $1.552\times 10^{-4}$ | $0.00000258 \to 0.00000278$ |
| **10** | 4.4132 | True | 8192 $\to$ 8192 | **False** | $1.665\times 10^{-4}$ | $0.00000278 \to 0.00000298$ |

**Summary:** 10/10 optimizer steps executed cleanly. Gradients remained strictly finite, no step was skipped, and parameter weights changed by positive norms on every single step.

**Full Test Suite Status:** **32/32 unit tests passing** via `tests/run_all_tests.py`.

---

## 5. Execution Commands for Phase 6B Training Runs

The training pipeline is fully encapsulated in `scripts/train_phase6b.py`. Below are the four exact execution commands:

### Command for Run A (Baseline MIL-NCE + Cosine LR Schedule)
```powershell
.\venv\Scripts\python scripts/train_phase6b.py --run_id run_a --init_checkpoint checkpoints/spectrum_molecule_step_5000.pt --total_steps 5000 --warmup_steps 500 --checkpoint_interval 1000
```
- **Configuration:** Objective = `multi_positive`, Negative Pool = Disabled, Scheduler = Cosine Warmup ($10^{-4} \to 10^{-6}$).
- **Output Directory:** `checkpoints/phase6b_run_a/` (`step_1000.pt` ... `step_5000.pt`)

### Command for Run B (Normalized SupCon + Cosine LR Schedule)
```powershell
.\venv\Scripts\python scripts/train_phase6b.py --run_id run_b --init_checkpoint checkpoints/spectrum_molecule_step_5000.pt --total_steps 5000 --warmup_steps 500 --checkpoint_interval 1000
```
- **Configuration:** Objective = `normalized_supcon`, Negative Pool = Disabled, Scheduler = Cosine Warmup ($10^{-4} \to 10^{-6}$).
- **Output Directory:** `checkpoints/phase6b_run_b/` (`step_1000.pt` ... `step_5000.pt`)

### Command for Run C (Baseline MIL-NCE + Cosine LR + Chemistry-Aware Negatives)
```powershell
.\venv\Scripts\python scripts/train_phase6b.py --run_id run_c --init_checkpoint checkpoints/spectrum_molecule_step_5000.pt --total_steps 5000 --warmup_steps 500 --checkpoint_interval 1000 --negatives_per_anchor 2
```
- **Configuration:** Objective = `multi_positive`, Negative Pool = Enabled (Mass + Tanimoto + Random), Scheduler = Cosine Warmup ($10^{-4} \to 10^{-6}$).
- **Output Directory:** `checkpoints/phase6b_run_c/` (`step_1000.pt` ... `step_5000.pt`)

### Command for Run D (Normalized SupCon + Cosine LR + Chemistry-Aware Negatives)
```powershell
.\venv\Scripts\python scripts/train_phase6b.py --run_id run_d --init_checkpoint checkpoints/spectrum_molecule_step_5000.pt --total_steps 5000 --warmup_steps 500 --checkpoint_interval 1000 --negatives_per_anchor 2
```
- **Configuration:** Objective = `normalized_supcon`, Negative Pool = Enabled (Mass + Tanimoto + Random), Scheduler = Cosine Warmup ($10^{-4} \to 10^{-6}$).
- **Output Directory:** `checkpoints/phase6b_run_d/` (`step_1000.pt` ... `step_5000.pt`)

---

## 6. Post-Training Evaluation Protocol

Following training completion:
1. Each run's terminal checkpoint (`step_5000.pt`) and best intermediate checkpoint will be evaluated against the candidate bank of 277,566 SMILES across the 400 held-out validation molecules.
2. Ranking strategy: Mass-Aware Mean scoring at $\sigma = 20$ ppm.
3. Reported metrics: MRR@25, Recall@1, Recall@5, Recall@10, Recall@25, Mean Rank, and Median Rank.
4. Champion comparison: Any run achieving MRR@25 $> 0.2686$ and Recall@25 $> 69.75\%$ will be flagged as the new candidate champion.
5. If a new champion is identified, full submission predictions for the 400 test molecules (1,213 spectra) will be re-generated and verified against submission integrity criteria (400 rows, exactly 25 SMILES per row, zero NaNs).
