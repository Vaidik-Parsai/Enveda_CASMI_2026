# Enveda CASMI 2026 Pipeline Audit

## 1. Executive Summary
This document provides a comprehensive technical audit of the Enveda CASMI 2026 mass-spectrometry-to-molecule retrieval system. It identifies key memory bottlenecks, potential data leakage risks in the exploratory notebook workflow, verifies checkpoint compatibility requirements, and outlines the production-ready modular architecture.

---

## 2. Dataset Size & Schema Findings

### 2.1 Dataset Volume
- **Total Training Spectra:** 2,539,608 rows
- **Unique Molecules (Normalized SMILES):** 277,566
- **Unique Molecule Identities (`inchikey14`):** 275,810
- **Validation Spectra (5% group split):** 130,498 rows
- **Training Spectra (95% group split):** 2,409,110 rows

### 2.2 Schema Definition
- **Training Columns (`train.parquet`):**
  - Identifiers & Molecular annotations: `ingest_lib`, `normalized_smiles`, `inchikey`, `inchikey14`, `molecular_formula`
  - Metadata features: `ionization_mode`, `instrument_type`, `adduct`, `adduct_orig`, `precursor_mz`, `precursor_error_ppm`, `base_peak_intensity`, `collision_energy_ev`, `collision_energy_orig`, `collision_energy_orig_units`
  - MS2 Spectrum arrays: `ms2_mzs` (list of float m/z), `ms2_normalized_intensities` (list of float intensities), `num_peaks` (int)
- **Test Columns (`test.parquet`):**
  - Identifiers: `spectrum_id`, `molecule_id` (candidate queries)
  - Metadata features: `ionization_mode`, `instrument_type`, `adduct`, `precursor_mz`, `base_peak_intensity`, `collision_energy_ev`, `collision_energy_orig`, `collision_energy_orig_units`
  - MS2 Spectrum arrays: `ms2_mzs`, `ms2_normalized_intensities`

---

## 3. Current Architecture & Compatibility Requirements

### 3.1 Two-Tower Model Specifications
- **Spectrum Encoder (`SpectrumEncoder`):**
  - Input: Binned spectrum vector (10,000 bins: 0.1 Da bins up to 1000 m/z) concatenated with tabular metadata features (130 dims). Total input dimension = **10,130**.
  - Layers:
    - `Linear(10130, 512)` -> `LayerNorm(512)` -> `GELU()` -> `Dropout(0.2)`
    - `Linear(512, 256)` -> `LayerNorm(256)` -> `GELU()` -> `Dropout(0.1)`
    - `Linear(256, 128)`
  - Output: L2-normalized 128-dimensional embedding ($||x||_2 = 1.0$).
- **Molecule Encoder (`MoleculeEncoder`):**
  - Input: Morgan fingerprint from `normalized_smiles` (Radius=2, `n_bits=2048`). Total input dimension = **2,048**.
  - Layers:
    - `Linear(2048, 512)` -> `LayerNorm(512)` -> `GELU()` -> `Dropout(0.2)`
    - `Linear(512, 256)` -> `LayerNorm(256)` -> `GELU()` -> `Dropout(0.1)`
    - `Linear(256, 128)`
  - Output: L2-normalized 128-dimensional embedding ($||x||_2 = 1.0$).
- **Two-Tower Wrapper (`SpectrumMoleculeModel`):**
  - Hosts `self.spectrum_encoder` and `self.molecule_encoder`.
  - In forward pass, returns `(spectrum_embeddings, molecule_embeddings)`.
- **Loss Function:**
  - Symmetric Multi-Positive Contrastive Loss (`multi_positive_contrastive_loss`).
  - Cosine similarity logits with temperature `temperature = 0.07`.
  - Groups identical molecules in-batch using `molecule_ids` mask (`inchikey14` factorized integer ID).
  - Uses `logsumexp` over positive and all pairs symmetrically: $0.5 \times (\mathcal{L}_{spec \to mol} + \mathcal{L}_{mol \to spec})$.
- **Optimizer:** `AdamW(lr=1e-4, weight_decay=1e-4)`, gradient clipping `max_norm=1.0`.

### 3.2 Checkpoint Compatibility Requirements
Existing checkpoints:
- `checkpoints/spectrum_molecule_step_100.pt`
- `checkpoints/spectrum_molecule_step_1000.pt`

Both store:
- `step` (int)
- `model_state_dict` (dict)
- `optimizer_state_dict` (dict)
- `loss_history` (list)

State dict naming hierarchy:
- `spectrum_encoder.network.{0,1,4,5,8}.{weight,bias}`
- `molecule_encoder.network.{0,1,4,5,8}.{weight,bias}`

**Strict rule:** The refactored model class must strictly preserve these exact module names, layer indices, and tensor shapes so that existing checkpoints can be loaded without modification or translation.

---

## 4. Memory Bottlenecks & Vulnerabilities in Existing Notebook

1. **Monolithic Dataframe Ingestion (`pd.read_parquet(train.parquet)`):**
   - Attempting to load 2,539,608 rows with variable-length list columns (`ms2_mzs`, `ms2_normalized_intensities`) consumes ~15–25 GB of system RAM.
2. **Dense Python Iteration (`df.apply(filter_spectrum_advanced)`):**
   - Applies Python functions row-by-row on millions of rows, creating temporary Python lists and putting extreme pressure on garbage collection.
3. **Full Dataset Concatenation & Sparse Matrix Creation:**
   - `np.concatenate(df['ms2_mzs'].to_numpy())` materializes hundreds of millions of elements in RAM simultaneously.
   - `X_train_sparse` (2.54M x 10,000) occupies significant memory, and random row indexing into CSR matrices inside training loops causes random pointer chasing.
4. **Data Leakage in Exploratory Feature Fitting:**
   - In `preprocessing.ipynb`, `ColumnTransformer` (`StandardScaler` and `OneHotEncoder`) was fitted on the entire dataset `df_tr` before performing the `GroupShuffleSplit`. This leaks validation distribution statistics into training.
5. **Evaluation OOM Catastrophe Risk:**
   - Computing a full matrix product between validation spectrum embeddings ($N_{val} \approx 130,498$) and the candidate bank ($N_{cand} \approx 277,566$) would require an intermediate matrix of $130,498 \times 277,566 \approx 36.2 \times 10^9$ floats ($\approx 145 \text{ GB}$), causing immediate memory exhaustion on any workstation or 6GB GPU.

---

## 5. Target Refactoring Plan

### Phase 2: Data Pipeline
- **Streaming / Chunked Reading:** Utilize PyArrow's `ParquetFile.iter_batches()` and chunk-based processing to bound memory footprint.
- **Precomputed Chunk / Shard Artifacts:** Optionally pre-bin or process Parquet row groups into disk-backed shards for fast, low-RAM training.
- **Centralized Preprocessor:**
  - Pure, stateless vectorization functions (`filter_spectrum`, `bin_spectrum`, `extract_ce_stats`).
  - Fit `StandardScaler` and `OneHotEncoder` strictly on training data; persist artifacts to disk.
- **Deterministic Group Splitting:**
  - Group-split based on `inchikey14` with a fixed seed (42) and configurable fraction (0.05).
  - Save split indices/manifest to disk.

### Phase 3: Model and Training
- Package model into `src/enveda/models/model.py` guaranteeing exact parameter naming and shapes.
- Implement reusable batch trainer in `src/enveda/training/trainer.py` with:
  - Bounded memory mini-batches.
  - Automatic Mixed Precision (`torch.autocast(device_type="cuda", dtype=torch.float16)`).
  - Step & epoch tracking, running loss averages, memory logging.
  - Safe checkpointing preserving existing format and adding best model tracking.

### Phase 4: Memory-Bounded Retrieval & Evaluation
- **Candidate Bank:** Precompute candidate molecule embeddings in bounded chunks and store on disk (e.g., memory-mapped `.npy` or PyTorch binary chunks).
- **Chunked Top-K Search:** Evaluate validation query embeddings against candidate blocks in chunks (e.g. 10,000 candidates per step), maintaining running Top-25 candidates without ever materializing the $N_{val} \times N_{cand}$ matrix.
- **Metrics:** Compute Recall@1, Recall@5, Recall@25, and Mean Reciprocal Rank (MRR).

### Phase 5: Verification & CLI
- Comprehensive unit tests covering all components.
- Lightweight end-to-end smoke test script runnable via PowerShell.
- Updated documentation and README.
