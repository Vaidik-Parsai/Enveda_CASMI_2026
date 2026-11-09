# Enveda CASMI 2026: Mass-Spectra-to-Molecule Retrieval Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09+-green.svg)](https://www.rdkit.org/)
[![Tests Passing](https://img.shields.io/badge/tests-32%2F32%20passing-brightgreen.svg)](tests/)
[![MRR@25 Champion](https://img.shields.io/badge/Official%20Champion%20MRR@25-0.2686-gold.svg)](docs/CHAMPION_MODEL.md)
[![Recall@25 Record](https://img.shields.io/badge/Record%20Recall@25-70.50%25-brightgreen.svg)](docs/PHASE_6B_RUN_C_RESULTS.md)

Production-grade, modular, and memory-efficient deep learning pipeline for the **Enveda CASMI 2026** competition: identifying 2D molecular structures from experimental tandem mass spectrometry ($\text{MS/MS}$) data.

---

## 1. Executive Summary & Official Champion Benchmark

In the Enveda CASMI 2026 challenge, algorithms must identify the true chemical structure for unknown molecules from their experimental mass spectra, selecting from an open candidate database of **277,566 candidate SMILES**.

### Official Champion Benchmark
* **Champion Model Checkpoint:** [`checkpoints/spectrum_molecule_step_5000.pt`](docs/CHAMPION_MODEL.md)
* **Primary Competition Metric:** **MRR@25 = 0.2686**
* **Validation Performance (400 Held-Out Molecules / 17,267 Spectra):**
  * **Recall@1:** **16.00%**
  * **Recall@5:** **40.00%**
  * **Recall@10:** **50.50%**
  * **Recall@25:** **69.75%**
  * **Mean Rank:** **12.89** | **Median Rank:** **10.0**
  * **Target Survival Rate:** **98.00%** ($\pm 50\text{ ppm}$ consensus neutral mass ceiling)
  * **Effective Rank ($\text{Rank}_{\text{eff}}$):** **70.57** (Optimal representation isotropy)
* **Physics-Informed Mass Scoring Breakthrough:** Combining dual-tower neural cosine similarity with Gaussian soft precursor neutral mass scoring ($\sigma = 20\text{ ppm}$) improved MRR@25 from **0.0006 to 0.2686** (**+44,667% relative improvement**).
* **New Project Record for Candidate Coverage:** In Phase 6B Run C, chemistry-conditioned hard-negative training pushed **Recall@25 to an all-time record 70.50%**.
* **Zero Leakage & Reproducibility:** 100% deterministic scaffold grouping on InChIKey14, strict candidate identity masking, AMP mixed-precision training (`init_scale=8192.0`), and **32/32 passing unit tests**.

---

## 2. End-to-End System Architecture

```
[ Query MS/MS Spectrum ]                       [ Candidate Molecular Structures ]
 (m/z, intensity arrays)                               (277,566 SMILES)
           │                                                   │
  Noise Thresholding &                                 RDKit Canonicalization &
  Top-128 Peak Selection                               Morgan Fingerprints (2048-bit)
           │                                                   │
  Sparse m/z Binning (10,000 bins)                             │
           │                                                   │
  Tabular Metadata (130 dims)                                  │
  (Precursor m/z, Adduct, Mode, CE)                            │
           │                                                   │
           ▼                                                   ▼
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│     Spectrum Encoder (MLP)      │                 │     Molecule Encoder (MLP)      │
│  Linear(10130 -> 512), LN, GELU │                 │  Linear(2048 -> 512), LN, GELU  │
│  Linear(512 -> 256),   LN, GELU │                 │  Linear(512 -> 256),   LN, GELU │
│  Linear(256 -> 128),   L2-Norm  │                 │  Linear(256 -> 128),   L2-Norm  │
└─────────────────────────────────┘                 └─────────────────────────────────┘
                 │                                                   │
                 ▼                                                   ▼
         Spectrum Embedding                                 Molecule Embedding
           z_s in S^127                                        z_m in S^127
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                               Neural Cosine Similarity
                                 S_neural = <z_s, z_m>
                                           │
[ Experimental Precursor m/z ]             │
              │                            │
   Adduct Mass Compensation                │
              │                            │
   Consensus Neutral Mass                  │
              │                            │
   Gaussian Soft Mass Scoring ─────────────┤
    S_mass = exp(-0.5*(ppm/20)^2)          ▼
                             Combined Mass-Aware Score
                          S_comb = S_neural * S_mass
                                           │
                                           ▼
                             Multi-Spectrum Aggregation
                         (Mass-Aware Mean across spectra)
                                           │
                                           ▼
                             Top-25 Ranked SMILES Output
```

---

## 3. Comprehensive Experimental Benchmarks

All evaluations are conducted against the full **277,566 candidate SMILES bank** across **400 held-out validation molecules** (17,267 experimental spectra) under identical conditions ($\sigma = 20\text{ ppm}$):

| Experiment / Checkpoint | Description | MRR@25 | Recall@1 | Recall@5 | Recall@10 | Recall@25 | Target Survival | Effective Rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Raw Baseline** | Neural similarity only (no mass) | 0.0006 | 0.00% | 0.00% | 0.25% | 0.50% | 100.00% | 78.1 |
| **Mass-Only Baseline** | Gaussian soft mass only (no neural) | 0.0435 | 1.75% | 5.50% | 8.75% | 18.25% | 98.00% | — |
| **Current Champion (`step_5000.pt`)** | Multi-positive MIL-NCE (in-batch) | **0.2686** | **16.00%** | **40.00%** | 50.50% | 69.75% | **98.00%** | **70.6** |
| **Step 10,000** | Constant LR training drift | 0.2638 | 15.50% | 39.25% | 49.75% | 69.00% | 98.00% | 64.2 |
| **Step 20,000** | Constant LR training drift | 0.2604 | 15.00% | 38.50% | 49.00% | 68.50% | 98.00% | 55.3 |
| **Phase 6B Run A (Step 5,000)** | Cosine LR Warmup Baseline | 0.2573 | 15.00% | 36.75% | **51.50%** | 67.75% | 98.00% | 66.1 |
| **Phase 6B Run C (Step 2,000)** | MIL-NCE + Chemistry Negatives | 0.2525 | 14.00% | 37.25% | **51.50%** | **70.50%** | **98.00%** | 63.3 |

### Metric Definitions (Audited in `docs/CHAMPION_MODEL.md`)
* **Target Survival Rate (98.00%):** The proportion of true target molecules whose monoisotopic neutral mass falls within the candidate pool's consensus neutral mass tolerance ($\pm 50\text{ ppm}$). This represents the theoretical upper-bound ceiling for mass-filtered retrieval.
* **Recall@25 (70.50% Record):** The proportion of true target molecules successfully retrieved and ranked inside the top 25 candidates by our dual-tower scoring engine.

### Data Utilisation Provenance (Audited in `docs/DATA_UTILIZATION_AUDIT.md`)
* **Total Training Split Rows:** 2,409,110 spectra.
* **Presentations Consumed in 5,000 Steps:** 640,000 spectra (**0.266 epochs**, **26.57%** of training split, representing 103,220 unique molecules).
* **Sample Efficiency:** Achieving MRR@25 = 0.2686 on just 0.27 epochs highlights the high sample efficiency of the dual-tower architecture.

---

## 4. Documentation Map

| Document | Purpose & Key Takeaways |
|---|---|
| [`docs/CHAMPION_MODEL.md`](docs/CHAMPION_MODEL.md) | **Official Champion Specification:** Exact weights, hyperparameters, metric definitions, and rejection criteria for alternative runs. |
| [`docs/DATA_UTILIZATION_AUDIT.md`](docs/DATA_UTILIZATION_AUDIT.md) | **Data Ingestion Audit:** Demonstrates that 640,000 unique presentations (26.57% of split) were consumed across 103,220 unique molecules. |
| [`docs/TEST_DATA_AUDIT.md`](docs/TEST_DATA_AUDIT.md) | **Competition Test Exploration:** Schema audit of 1,213 test spectra across 400 query molecules, adduct distributions, and zero-null verification. |
| [`docs/TEST_INFERENCE_AND_SUBMISSION.md`](docs/TEST_INFERENCE_AND_SUBMISSION.md) | **Inference & Submission Guide:** 400 molecules $\times$ 25 semicolon-delimited SMILES verified against competition sample format. |
| [`docs/FINAL_REPOSITORY_AUDIT.md`](docs/FINAL_REPOSITORY_AUDIT.md) | **Full Repository Inventory:** 12-category structural audit of every file, dependency references, and cleanup strategy. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **System Architecture:** Mathematical formulation of encoders, binning, metadata scaling, mass filtering, and Mermaid diagrams. |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | **Experimental Progression:** Detailed history from Phase 1 through Phase 6B, ablation benchmarks, and $\sigma$ parameter tuning. |
| [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) | **Technical Interview Guide:** 18 multi-level questions answered at intuitive, technical, and applied levels. |
| [`docs/LIMITATIONS_AND_FUTURE_WORK.md`](docs/LIMITATIONS_AND_FUTURE_WORK.md) | **Limitations & Research Directions:** Sample size confidence bounds, closed library constraints, and proposed GNN/Transformer architectures. |

---

## 5. Repository Structure

```text
Enveda CASMI/
├── configs/
│   └── default.yaml                 # Centralized configuration (paths, hyperparams)
├── docs/                            # Complete technical documentation suite (16 reports)
├── src/
│   └── enveda/
│       ├── config.py                # Configuration loader
│       ├── data/
│       │   ├── dataset.py           # StreamingParquetDataset and batch processing
│       │   ├── negative_sampler.py  # ChemistryAwareNegativePool (mass, structural, random)
│       │   ├── preprocessing.py     # Noise filtering, m/z binning, Morgan fingerprints
│       │   ├── features.py          # Tabular metadata scaling & categorical encoding
│       │   └── split.py             # Deterministic GroupShuffleSplit by inchikey14
│       ├── models/
│       │   └── model.py             # SpectrumEncoder, MoleculeEncoder, SpectrumMoleculeModel
│       ├── training/
│       │   ├── losses.py            # MIL-NCE & Normalized Supervised Contrastive Loss
│       │   ├── schedulers.py        # CosineWarmupScheduler with minimum floor
│       │   ├── diagnostics.py       # Effective rank and embedding isotropy tracking
│       │   ├── checkpointing.py     # Backward-compatible checkpoint persistence
│       │   └── trainer.py           # AMP mixed-precision batch trainer (init_scale=8192)
│       ├── evaluation/
│       │   ├── mass_filter.py       # Consensus neutral mass & Gaussian soft mass scoring
│       │   ├── aggregation.py       # Multi-spectrum molecule-level score fusion
│       │   ├── reranker.py          # CandidateReranker orchestration
│       │   ├── metrics.py           # MRR@25, Recall@1/5/10/25, Mean/Median Rank
│       │   └── retrieval.py         # Chunked GPU top-k candidate retrieval
│       └── utils/
│           ├── logging.py           # Dual console & file logger
│           └── reproducibility.py   # Deterministic seed management
├── scripts/
│   ├── predict.py                   # Master test prediction & submission generator
│   ├── evaluate.py                  # Chunked evaluation on validation queries
│   ├── train_phase6b.py             # Phase 6B controlled experiment runner
│   ├── build_candidate_bank.py      # Precompute candidate embeddings & neutral masses
│   └── run_mass_ablation_and_step_eval.py # Ablation benchmark runner
├── tests/
│   ├── test_preprocessing.py        # Peak filtering and fingerprint tests
│   ├── test_model.py                # Model shapes, gradient flow, and checkpoint tests
│   ├── test_mass_filter.py          # Exact monoisotopic mass & soft scoring tests
│   ├── test_aggregation.py          # Molecule-level score aggregation tests
│   ├── test_chemistry_aware_training.py # SupCon gradients, scheduler & sampler tests
│   └── run_all_tests.py             # Dependency-free master test runner
├── checkpoints/                     # Model weights (.pt files, excluded from Git)
├── outputs/                         # Candidate banks and submission files (excluded)
├── requirements.txt
└── README.md
```

---

## 6. Getting Started & Reproducibility

### 6.1 Environment Setup
```powershell
# 1. Clone repository
git clone https://github.com/Vaidik-Parsai/Enveda_CASMI_2026.git
cd "Enveda CASMI"

# 2. Create and activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### 6.2 Execute the Unit Test Suite
To verify the complete mathematics, gradient backpropagation, and data preprocessing:
```powershell
.\venv\Scripts\python tests/run_all_tests.py
# Output: TEST SUMMARY: 32/32 Passed, 0 Failed
```

### 6.3 Building the Candidate Bank
Precompute 128-dimensional embeddings and exact monoisotopic neutral masses for all 277,566 candidate SMILES:
```powershell
.\venv\Scripts\python scripts/build_candidate_bank.py --checkpoint checkpoints/spectrum_molecule_step_5000.pt
```

### 6.4 Evaluating the Champion Checkpoint
Run the official evaluation across the 400 validation molecules using mass-aware mean scoring:
```powershell
.\venv\Scripts\python scripts/evaluate.py --checkpoint checkpoints/spectrum_molecule_step_5000.pt --sigma 20.0
```

### 6.5 Generating Competition Test Predictions
Generate the official CASMI top-25 ranked predictions for the 400 test molecules (1,213 spectra):
```powershell
.\venv\Scripts\python scripts/predict.py `
    --checkpoint checkpoints/spectrum_molecule_step_5000.pt `
    --method mass_mean `
    --sigma_ppm 20.0 `
    --top_k 25 `
    --output_csv outputs/submission_top25.csv
```
This generates `outputs/submission_top25.csv` and mirrors to `outputs/submission.csv` (400 rows, 25 semicolon-delimited SMILES, strictly matching the competition schema).
