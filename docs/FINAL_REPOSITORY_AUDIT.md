# Enveda CASMI 2026: Final Comprehensive Repository Audit

**Document ID:** `docs/FINAL_REPOSITORY_AUDIT.md`  
**Purpose:** Exhaustive structural inventory and lifecycle classification of all repository files prior to final cleanup and test inference.  
**Auditor:** Senior ML Research Engineer (Mass Spectrometry, Cheminformatics, Retrieval)  

---

## 1. Inventory & Lifecycle Classification

Each file is classified by its functional category, purpose, dependency references, necessity for reproducing the final champion retrieval pipeline, and safety for removal.

### Category 1: Core Source Code & Utilities
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `src/enveda/__init__.py` | Package root export | All internal modules | Yes | No |
| `src/enveda/config.py` | Typed dataclass YAML loader for runtime configuration | `train.py`, `evaluate.py`, `predict.py`, `train_phase6b.py` | Yes | No |
| `src/enveda/utils/__init__.py` | Utilities module export (`setup_logger`, `set_seed`) | All training and eval scripts | Yes | No |
| `src/enveda/utils/logging.py` | Formatted dual console and disk logger | `trainer.py`, `retrieval.py`, `evaluate.py`, `train_phase6b.py` | Yes | No |
| `src/enveda/utils/reproducibility.py` | Deterministic random seed management (torch, numpy, cuda) | `trainer.py`, `train.py`, `train_phase6b.py` | Yes | No |

### Category 2: Data Preprocessing & Ingestion
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `src/enveda/data/__init__.py` | Preprocessing and dataset exports | `train.py`, `evaluate.py`, `predict.py` | Yes | No |
| `src/enveda/data/preprocessing.py` | Peak noise filtering, top-128 selection, 10,000-bin sparse vectorization, RDKit Morgan fingerprints | `dataset.py`, `predict.py`, `evaluate.py`, `build_candidate_bank.py` | Yes | No |
| `src/enveda/data/features.py` | Tabular metadata normalization and one-hot encoding (130 dims) | `dataset.py`, `predict.py`, `evaluate.py` | Yes | No |
| `src/enveda/data/split.py` | Deterministic GroupShuffleSplit on `inchikey14` (0% scaffold leakage) | `train.py`, `evaluate.py`, `train_phase6b.py` | Yes | No |
| `src/enveda/data/dataset.py` | StreamingParquetDataset and memory-mapped batch iterators | `trainer.py`, `train.py`, `train_phase6b.py` | Yes | No |
| `src/enveda/data/collate.py` | PyTorch DataLoader collation functions | `dataset.py` | Yes | No |

### Category 3: Model Architecture
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `src/enveda/models/__init__.py` | Model architecture exports | Core scripts | Yes | No |
| `src/enveda/models/model.py` | `SpectrumEncoder` (10130->128 MLP), `MoleculeEncoder` (2048->128 MLP), `SpectrumMoleculeModel` | `trainer.py`, `evaluate.py`, `predict.py`, `build_candidate_bank.py` | Yes | No |

### Category 4: Training & Optimization
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `src/enveda/training/__init__.py` | Training package exports | Training runners | Yes | No |
| `src/enveda/training/losses.py` | Multi-positive MIL-NCE and Normalized Supervised Contrastive Loss | `trainer.py`, `train_phase6b.py`, `tests` | Yes | No |
| `src/enveda/training/trainer.py` | AMP mixed-precision trainer with GradScaler and periodic checkpointing | `train.py`, `train_phase6b.py` | Yes | No |
| `src/enveda/training/checkpointing.py` | State-dict checkpoint saver and loader with scheduler/scaler preservation | `trainer.py`, `train_phase6b.py`, `evaluate.py`, `predict.py` | Yes | No |
| `src/enveda/training/schedulers.py` | `CosineWarmupScheduler` with linear warmup and cosine floor | `train_phase6b.py`, `trainer.py`, `tests` | Yes | No |
| `src/enveda/training/diagnostics.py` | Effective rank ($\text{Rank}_{\text{eff}}$) and isotropy tracker | `trainer.py`, `tests` | Yes | No |
| `src/enveda/data/negative_sampler.py` | `ChemistryAwareNegativePool` for mass-compatible and structural negatives | `train_phase6b.py`, `tests` | Yes | No |
| `scripts/train.py` | Standard baseline training CLI | Standalone CLI | Yes | No |
| `scripts/train_phase6b.py` | Phase 6B multi-arm training runner (Runs A, B, C, D) | Experiment runner | Yes | No |

### Category 5: Evaluation & Reranking
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `src/enveda/evaluation/__init__.py` | Evaluation package exports | Evaluation scripts | Yes | No |
| `src/enveda/evaluation/metrics.py` | Truncated MRR@25, Recall@1/5/10/25, Mean/Median Rank | `evaluate.py`, `evaluate_phase6b_*.py` | Yes | No |
| `src/enveda/evaluation/retrieval.py` | Chunked GPU matrix dot product retrieval | `evaluate.py`, `predict.py` | Yes | No |
| `src/enveda/evaluation/mass_filter.py` | Consensus neutral mass estimation and Gaussian soft mass scoring ($\sigma=20\text{ ppm}$) | `evaluate.py`, `predict.py`, `reranker.py` | Yes | No |
| `src/enveda/evaluation/aggregation.py` | Multi-spectrum molecule-level score aggregation (Mean, Max, RRF) | `evaluate.py`, `predict.py`, `reranker.py` | Yes | No |
| `src/enveda/evaluation/reranker.py` | Modular `CandidateReranker` class | `benchmark_reranker_and_aggregation.py` | Yes | No |
| `scripts/evaluate.py` | Master validation evaluation CLI for any checkpoint | Validation pipeline | Yes | No |
| `scripts/evaluate_phase6b_run_a.py` | Evaluator for Run A checkpoints | Phase 6B reproducibility | Yes | No |
| `scripts/evaluate_phase6b_run_c.py` | Evaluator for Run C checkpoints | Phase 6B reproducibility | Yes | No |
| `scripts/run_mass_ablation_and_step_eval.py` | Ablation benchmark runner across mass sigmas and checkpoints | Ablation reproducibility | Yes | No |

### Category 6: Inference & Submission
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `scripts/build_candidate_bank.py` | Precomputes 277,566 candidate embeddings and exact neutral masses | Candidate bank initialization | Yes | No |
| `scripts/predict.py` | Generates top-25 candidate rankings for test spectra/molecules | Competition submission pipeline | Yes | No |

### Category 7: Configuration & Environment
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `configs/default.yaml` | Master project YAML configuration | All scripts | Yes | No |
| `requirements.txt` | Python library dependencies | Environment setup | Yes | No |
| `.gitignore` | Excludes weights, outputs, caches, datasets | Git repository safety | Yes | No |

### Category 8: Unit Tests
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `tests/run_all_tests.py` | Master standalone test suite runner | Test execution | Yes | No |
| `tests/test_preprocessing.py` | Tests for peak filtering, binning, and Morgan fingerprints | Test suite | Yes | No |
| `tests/test_model.py` | Tests for shapes, gradient backprop, and checkpoint save/load | Test suite | Yes | No |
| `tests/test_dataset_and_split.py` | Tests for group split zero-overlap and metadata preprocessor | Test suite | Yes | No |
| `tests/test_retrieval.py` | Tests for chunked top-k retrieval vs brute-force | Test suite | Yes | No |
| `tests/test_aggregation.py` | Tests for molecule-level Mean, Max, and RRF aggregation | Test suite | Yes | No |
| `tests/test_mass_filter.py` | Tests for exact monoisotopic mass and Gaussian soft scoring | Test suite | Yes | No |
| `tests/test_chemistry_aware_training.py`| Tests for Normalized SupCon, scheduler, and negative sampler | Test suite | Yes | No |
| `tests/__init__.py` | Test package marker | Test imports | Yes | No |

### Category 9: Documentation
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `README.md` | Primary entry point with Mermaid architecture and benchmarks | Project documentation | Yes | No |
| `docs/ARCHITECTURE.md` | Comprehensive system architecture and data-flow diagrams | Architectural reference | Yes | No |
| `docs/EXPERIMENTS.md` | Complete experimental history and ablation benchmarks | Experimental reference | Yes | No |
| `docs/INTERVIEW_GUIDE.md` | 18 multi-level technical interview questions and answers | Interview preparation | Yes | No |
| `docs/LIMITATIONS_AND_FUTURE_WORK.md`| System limitations, statistical bounds, and future research | Technical reference | Yes | No |
| `docs/PHASE_6B_RUN_C_RESULTS.md` | Empirical report for Run C execution and benchmark | Experiment report | Yes | No |
| `docs/PHASE_6B_RUN_A_RESULTS.md` | Empirical report for Run A execution and benchmark | Experiment report | Yes | No |
| `docs/PHASE_6B_EXPERIMENT_DESIGN.md`| Design document for controlled 4-arm experiment | Experiment reference | Yes | No |
| `docs/PHASE_6A_TRAINING_AUDIT.md` | Audit of 20,000-step training failure modes | Diagnostic reference | Yes | No |
| `docs/METRICS_AUDIT.md` | Mathematical audit of truncated MRR@25 and scoring | Metric reference | Yes | No |
| `docs/PROJECT_STATUS_AUDIT.md` | Initial baseline audit and component status | Status reference | Yes | No |
| `docs/RETRIEVAL_IMPROVEMENT_REPORT.md`| Phase 5 retrieval improvement and mass filter report | Status reference | Yes | No |
| `docs/TRAINING_COMPARISON.md` | Comparison across training step checkpoints | Diagnostic reference | Yes | No |
| `docs/AGGREGATION_COMPARISON.md` | Ablation comparison of Mean, Max, and RRF aggregation | Benchmark reference | Yes | No |
| `docs/MASS_FILTER_EXPERIMENT.md` | Initial experiments on hard vs soft mass filtering | Experiment reference | Yes | No |
| `docs/HARD_NEGATIVE_EXPERIMENT.md` | Initial exploratory hard-negative formulation | Historical reference | Yes | No |

### Category 10: Experimental & Audit Scripts (Preserved for Reproducibility)
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `scripts/audit_adduct_mass.py` | Audits adduct distribution and exact mass errors | Scientific audit | Historical/audit | No (Preserve) |
| `scripts/audit_negative_sampler_shortfalls.py`| Diagnoses shortfall causes in negative sampler | Scientific audit | Historical/audit | No (Preserve) |
| `scripts/audit_structural_negatives.py` | Measures empirical Morgan Tanimoto distributions | Scientific audit | Historical/audit | No (Preserve) |
| `scripts/audit_training_behaviour.py` | Tracks gradient norms, effective rank, and positive alignment | Scientific audit | Historical/audit | No (Preserve) |
| `scripts/benchmark_checkpoints_and_reranking.py` | Multi-checkpoint reranker benchmark | Benchmark | Historical/audit | No (Preserve) |
| `scripts/benchmark_reranker_and_aggregation.py` | Aggregation method benchmark | Benchmark | Historical/audit | No (Preserve) |
| `scripts/experiment_mass_filter.py` | Mass filter threshold exploration | Benchmark | Historical/audit | No (Preserve) |
| `scripts/verify_optimizer_updates_10steps.py` | Verifies GradScaler scaling and optimizer updates | Integration audit | Historical/audit | No (Preserve) |
| `scripts/verify_pretraining_integration.py` | Traces negative sampler integration end-to-end | Integration audit | Historical/audit | No (Preserve) |

### Category 11: Obsolete / Redundant One-Off Files
| File Path | Purpose | Referenced / Imported By | Needed for Pipeline? | Safe to Remove? |
|---|---|---|:---:|:---:|
| `outputs/scratch/smoke_test_ckpt.pt` (26.3 MB) | Temporary dummy weights from Phase 1 smoke test | None | No | **Yes (Candidate for removal)** |
| `outputs/scratch/candidate_bank/` | Temporary dummy candidate bank from smoke test | None | No | **Yes (Candidate for removal)** |
| `scripts/smoke_test.py` | Synthetic end-to-end smoke test generating dummy checkpoints | Standalone test | No (Covered by unit tests) | Optional (Preserve or prune) |
| `download_data.py` | 5-line kagglehub download script | None | No | Safe to preserve or prune |
| `notebooks/preprocessing.ipynb` | Exploratory notebook from initial repository setup | None | No | Safe to preserve |

### Category 12: Generated Artifacts & Model Checkpoints (Protected by .gitignore)
| File Path / Pattern | Purpose | Action / Status |
|---|---|---|
| `checkpoints/spectrum_molecule_step_5000.pt` | **Official Champion Model Checkpoint** | **Preserve & Protect** |
| `checkpoints/phase6b_run_a/` | Run A checkpoints (Steps 1000–5000) | Preserve locally (ignored by git) |
| `checkpoints/phase6b_run_c/` | Run C checkpoints (Steps 1000–5000) | Preserve locally (ignored by git) |
| `checkpoints/spectrum_molecule_step_*.pt` | Historical checkpoints (Steps 100–20,000) | Preserve locally (ignored by git) |
| `outputs/candidate_bank/` | Candidate bank embeddings (277,566 x 128) and exact masses | Core retrieval artifact |
| `outputs/*.json` | Evaluation results (phase6b_run_a, phase6b_run_c, metrics.json) | Preserved experiment records |
| `outputs/submission_top25.csv` | Existing test predictions | Will be audited/regenerated for Phase 6 |

---

## 2. Summary of Cleanup Strategy (Phase 2)
1. **Remove temporary scratch artifacts:** Delete `outputs/scratch/` (containing obsolete 26.3 MB `smoke_test_ckpt.pt` and dummy candidate bank).
2. **Preserve all experimental scripts, audits, and unit tests:** All 32 unit tests, audit scripts, and benchmark tools are retained to ensure 100% reproducibility.
3. **Preserve all checkpoints locally:** The champion and experimental checkpoints are preserved on disk and kept strictly excluded from git tracking.
