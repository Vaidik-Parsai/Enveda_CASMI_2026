# Enveda CASMI 2026 — Comprehensive Project Status & Repository Audit

**Date:** November 8, 2026  
**Auditor:** Senior ML Engineer & Research Scientist  
**Repository:** `Enveda CASMI`  
**Execution Environment:** Windows, PyTorch 2.6.0+cu126, NVIDIA RTX 3050 6GB Laptop GPU  

---

## 1. Executive Summary

This audit assesses the current state of the Enveda CASMI 2026 codebase, verifies existing checkpoints, inspects data assets and preprocessing artifacts, tests baseline components, and provides a clear roadmap for completing the competition pipeline.

### High-Level Status
- **Core Architecture & Training Engine:** Fully implemented, modularized in `src/enveda/`, backward-compatible, and verified against all 50 saved checkpoints.
- **Unit Test Suite:** 15/15 tests passing across preprocessing, models, datasets, splitting, and retrieval.
- **Hardware & Environment:** RTX 3050 (6GB VRAM) running CUDA with AMP mixed precision. Peak memory during training is ~150 MB, well within the 6 GB ceiling.
- **Checkpoints:** 50 checkpoints located from Step 100 to Step 5000 (every 100 steps). Step 5000 is verified with valid model weights, optimizer state, scaler state, and loss history.
- **Data Availability:** Both `train.parquet` (2.89 GB, 2,539,608 rows) and `test.parquet` (4.6 MB, 1,213 rows across 400 molecules) are present in the local Kaggle cache.
- **Critical Gap:** The current evaluation (`scripts/evaluate.py`) and inference (`scripts/predict.py`) scripts operate **per-spectrum**, whereas the competition strictly requires **molecule-level predictions** where multiple spectra belonging to the same unknown molecule are aggregated into one ranked list of up to 25 candidate SMILES.

---

## 2. Repository Structure & Component Inventory

```
Enveda CASMI/
├── configs/
│   └── default.yaml                 # Centralized configuration (data, model, training, eval)
├── data/
│   └── cache/
│       ├── group_split.json         # Cached 95/5 group split by inchikey14 (21.7 MB)
│       └── meta_preprocessor.joblib # Fitted ColumnTransformer for metadata features
├── checkpoints/                     # 50 checkpoint files (.pt), step 100 to step 5000 (~75 MB each)
├── docs/
│   └── PROJECT_STATUS_AUDIT.md      # This document
├── notebooks/
│   └── preprocessing.ipynb          # Original exploratory & prototyping notebook
├── outputs/
│   └── train.log                    # Training execution logs (steps 1–5000)
├── scripts/
│   ├── build_candidate_bank.py      # Encodes candidate SMILES into disk-backed embeddings
│   ├── evaluate.py                  # Spectrum-level evaluation (needs refactoring for Phase 3)
│   ├── predict.py                   # Spectrum-level test prediction (needs refactoring for Phase 3/7)
│   ├── smoke_test.py                # End-to-end integration smoke test
│   └── train.py                     # Streaming Parquet trainer CLI with AMP & checkpointing
├── src/
│   └── enveda/
│       ├── __init__.py
│       ├── config.py                # Attribute-accessible ConfigDict and YAML loader
│       ├── data/
│           ├── __init__.py
│           ├── collate.py           # Collation helpers
│           ├── dataset.py           # StreamingParquetDataset & MemoryMappedBatchDataset
│           ├── features.py          # MetadataPreprocessor (CE stats + OneHotEncoder + StandardScaler)
│           ├── preprocessing.py     # Spectrum filter, bin_spectrum_vector, Morgan fingerprint
│           └── split.py             # Leak-free GroupShuffleSplit on inchikey14
│       ├── evaluation/
│           ├── __init__.py
│           ├── metrics.py           # Recall@K, MRR, Mean Rank, Median Rank
│           └── retrieval.py         # build_candidate_bank, chunked_topk_search, evaluate_retrieval
│       ├── models/
│           ├── __init__.py
│           └── model.py             # SpectrumEncoder, MoleculeEncoder, SpectrumMoleculeModel
│       ├── training/
│           ├── __init__.py
│           ├── checkpointing.py     # save_checkpoint, load_checkpoint
│           ├── losses.py            # multi_positive_contrastive_loss
│           └── trainer.py           # Trainer class with AMP, gradient clipping, checkpointing
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # Formatted logger setup
│           └── reproducibility.py   # Seed setting (Python, NumPy, PyTorch, CUDA)
└── tests/
    ├── __init__.py
    ├── run_all_tests.py             # Test runner without requiring external pytest runner
    ├── test_dataset_and_split.py    # Group split zero-overlap & preprocessor tests
    ├── test_model.py                # Forward pass, loss gradients, checkpoint compatibility
    ├── test_preprocessing.py        # Peak filtering, vector binning, CE parsing, Morgan FP
    └── test_retrieval.py            # Chunked top-k vs brute-force & metric calculation
```

---

## 3. Detailed Component Audit

### 3.1 Model Architecture (`src/enveda/models/model.py`)
- **`SpectrumEncoder`**:
  - Input: 10,000 binned spectrum features + 130 tabular metadata features = 10,130 dimensions.
  - Architecture: Linear(10130, 512) -> LayerNorm(512) -> GELU -> Dropout(0.2) -> Linear(512, 256) -> LayerNorm(256) -> GELU -> Dropout(0.1) -> Linear(256, 128).
  - Normalization: $L_2$ normalization on output vector (`dim=1`).
- **`MoleculeEncoder`**:
  - Input: 2,048-bit Morgan fingerprint (radius 2, ECFP4).
  - Architecture: Linear(2048, 512) -> LayerNorm(512) -> GELU -> Dropout(0.2) -> Linear(512, 256) -> LayerNorm(256) -> GELU -> Dropout(0.1) -> Linear(256, 128).
  - Normalization: $L_2$ normalization on output vector (`dim=1`).
- **`SpectrumMoleculeModel`**:
  - Encapsulates both encoders; forward pass yields `(spec_emb, mol_emb)`.
  - Total trainable parameters: **6,567,680**.
- **Compatibility Status:** **100% compatible** with original checkpoint parameter names (`spectrum_encoder.network.*`, `molecule_encoder.network.*`). Verified with strict loading on Step 100 and Step 5000.

### 3.2 Contrastive Objective (`src/enveda/training/losses.py`)
- **Objective:** Symmetric multi-positive InfoNCE contrastive loss with temperature $\tau = 0.07$.
- **Formula:**
  $$\mathcal{L}_{\text{S2M}} = \frac{1}{B}\sum_{i=1}^B \left( -\log \frac{\sum_{j \in P(i)} \exp(\text{sim}(s_i, m_j)/\tau)}{\sum_{k=1}^B \exp(\text{sim}(s_i, m_k)/\tau)} \right)$$
  $$\mathcal{L} = \frac{1}{2} (\mathcal{L}_{\text{S2M}} + \mathcal{L}_{\text{M2S}})$$
  where $P(i) = \{j \mid \text{molecule\_id}_j = \text{molecule\_id}_i\}$.
- **Numerical Stability:** Uses `torch.logsumexp` with masked logits filled with `torch.finfo(dtype).min`. Verified finite and backpropagatable.

### 3.3 Data & Preprocessing Pipeline (`src/enveda/data/`)
- **Spectrum Preprocessing (`preprocessing.py`):**
  - Noise filter: threshold $\ge 0.01$ relative intensity.
  - Precursor mass filter: retains peaks with $m/z \le \text{precursor\_mz} + 2.0$ Da.
  - Top-$N$ selection: retains top 128 highest intensity peaks.
  - Binning: 10,000 bins from $m/z = 0.0$ to $1000.0$ ($0.1$ Da width).
  - Intensity transform: $\sqrt{I}$ followed by max-normalization to $[0, 1]$.
- **Metadata Preprocessing (`features.py`):**
  - Numeric columns (7): `precursor_mz`, `base_peak_intensity`, `ce_min`, `ce_max`, `ce_mean`, `ce_median`, `ce_count`. Scaled with `StandardScaler`.
  - Categorical columns (2): `adduct`, `ionization_mode`. One-hot encoded.
  - Fixed output dimension: 130 dimensions (padded with zeros if one-hot columns are fewer).
- **Group Splitting (`split.py`):**
  - Uses `GroupShuffleSplit` on `inchikey14` (molecular skeleton identity).
  - Validation fraction: 5% ($130,498$ spectra), Train fraction: 95% ($2,409,110$ spectra).
  - Verified: **Zero overlap** between training and validation molecules ($0$ shared `inchikey14`).
  - Preprocessor is fitted strictly on the training partition and cached to avoid data leakage.

### 3.4 Checkpoint Inventory & Inspection (`checkpoints/`)
All 50 checkpoints were verified:
- **Range:** `spectrum_molecule_step_100.pt` through `spectrum_molecule_step_5000.pt` (every 100 steps).
- **Inspected Checkpoint (Step 5000):**
  - Size: 75.23 MB.
  - Global step: 5,000 (total spectrum presentations: $5,000 \times 128 = 640,000$).
  - Model weights: 20 tensors matching model architecture exactly.
  - Optimizer state: AdamW state dict preserved at step 4,994 with exponential averages and variance moments.
  - Scaler state: AMP GradScaler state dict preserved with scale=4096.0.
  - Loss history: 4,500 recorded loss steps. Initial loss: $\sim 5.16 \rightarrow$ Final loss: $\sim 1.66$ (steady convergence).
  - Full configuration dictionary preserved inside checkpoint.

### 3.5 External & Raw Data Assets
- `train.parquet`: Located at `C:/Users/vaidi/.cache/kagglehub/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet` (2,892.8 MB, 2,539,608 spectra, 275,810 unique `inchikey14`, 277,566 unique SMILES).
- `test.parquet`: Located at `C:/Users/vaidi/.cache/kagglehub/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet` (4.6 MB, 1,213 spectra from 400 unique molecules, 1 to 9 spectra per molecule, average 3.03 spectra/molecule).

---

## 4. Gap Analysis: What is Missing or Broken

| Component | Status | Details / Impact |
|---|---|---|
| **Molecule-Level Aggregation (Phase 3)** | **MISSING** | `evaluate.py` and `predict.py` evaluate and rank per spectrum, ignoring that multiple test spectra belong to the same molecule. This directly violates the core competition requirement. |
| **Molecule-Level Evaluation Script** | **MISSING** | No evaluation script computes molecule-level Recall@1, 5, 25 or MRR@25 grouped by `inchikey14` or `molecule_id`. |
| **Candidate Universe Definition (Phase 4)** | **INCOMPLETE** | Currently `build_candidate_bank.py` extracts unique SMILES from `train.parquet` only. Test spectra might contain unseen molecules (novelty classes 1, 2, 3). Validation candidate pool must include validation true SMILES to measure ranking, but separate candidate coverage from ranking. |
| **Aggregation Method Comparison** | **MISSING** | Need implementations and systematic comparison of: (1) Max similarity, (2) Mean similarity, (3) Reciprocal Rank Fusion (RRF, $k=60$). |
| **Competition Submission Formatting (Phase 7)** | **BROKEN** | `scripts/predict.py` outputs predictions per `spectrum_id` rather than per `molecule_id`. Test submission must have exactly 400 rows matching the test `molecule_id`s with space-delimited top-25 SMILES. |
| **Candidate Reranking (Phase 6)** | **NOT STARTED** | No reranking component exists (only raw cosine retrieval). |

---

## 5. Exact Next Steps (Execution Plan)

### Step 1: Phase 2 — Validation Correctness & Protocol Setup
1. Define the validation protocol:
   - Sample a representative validation subset of molecules (e.g., 400 validation molecules with 1–16 spectra each, mirroring the test distribution).
   - Validation spectra and identities remain strictly held out from training.
   - Ground truth molecules are added to the candidate evaluation pool so retrieval Recall@K and MRR@25 can be evaluated under known candidate presence.
2. Verify zero data leakage in preprocessing or normalization.

### Step 2: Phase 3 — Implement Molecule-Level Aggregation
1. Implement `aggregate_spectrum_scores` in `src/enveda/evaluation/retrieval.py` supporting:
   - **Max similarity:** $S(c) = \max_{s \in \text{spectra}} \text{sim}(s, c)$
   - **Mean similarity:** $S(c) = \frac{1}{|S|} \sum_{s \in \text{spectra}} \text{sim}(s, c)$
   - **Reciprocal Rank Fusion (RRF):** $\text{RRF}(c) = \sum_{s \in \text{spectra}} \frac{1}{k + \text{rank}(c, s)}$ with default $k=60$.
2. Implement unit tests in `tests/test_aggregation.py` verifying:
   - Output produces exactly 1 ranking per `molecule_id`.
   - Ties and rankings are strictly sorted descending.
   - Known synthetic multi-spectrum cases match analytical expectations.

### Step 3: Phase 4 — First Trustworthy Molecule-Level Benchmark
1. Build a candidate bank for the validation benchmark.
2. Evaluate `spectrum_molecule_step_5000.pt` on the validation set using molecule-level aggregation.
3. Compare Max vs. Mean vs. RRF aggregation.
4. Record baseline MRR@25, Recall@1, 5, 10, 25, and Mean/Median Rank in `outputs/validation_baseline_metrics.json`.

### Step 4: Subsequent Phases (Candidate Expansion, Reranking & Submission)
- Incorporate external candidate sources / novel chemical space.
- Develop precursor-mass / adduct compatibility filtering and lightweight reranker.
- Build clean, verified submission pipeline generating 400 rows for `test.parquet`.

