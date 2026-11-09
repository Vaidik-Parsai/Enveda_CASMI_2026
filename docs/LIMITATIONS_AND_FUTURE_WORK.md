# Enveda CASMI 2026: Limitations & Future Research Directions

**Document ID:** `docs/LIMITATIONS_AND_FUTURE_WORK.md`  
**Purpose:** Honest, scientifically rigorous appraisal of the current system's limitations, edge cases, failure modes, and defensible future directions.  

---

## 1. System Limitations & Known Vulnerabilities

### 1.1 Validation Set Sample Size & Statistical Uncertainty
- **Current Setup:** The validation set consists of $M = 400$ held-out unique molecular identities (encompassing 17,267 experimental spectra), partitioned via deterministic `GroupShuffleSplit` on 14-character InChIKey scaffolds.
- **Statistical Uncertainty:** With $M = 400$, each individual validation molecule represents $0.25\%$ ($1 / 400$) of the total recall. A $95\%$ binomial confidence interval for our champion's Recall@25 ($69.75\%$) spans approximately $[65.1\%, 74.1\%]$, and the bootstrap standard error for MRR@25 ($0.2686$) is approximately $\pm 0.016$. Small performance differences of $\Delta \text{MRR} < 0.010$ across experimental runs must therefore be interpreted with scientific caution.

### 1.2 Closed Candidate Library vs. De Novo Open-World Identification
- **Current Setup:** The retrieval system operates over a fixed candidate bank of $|\mathcal{C}| = 277,566$ unique SMILES.
- **Limitation:** In real-world natural products discovery (the core mission of Enveda Biosciences), novel secondary metabolites from uncharacterized plant or fungal clades often possess completely unprecedented chemical scaffolds that do not exist in any public database. If the true structure is absent from the candidate bank, retrieval-based ranking cannot output the correct structure, requiring complementary *de novo* molecular generation (e.g., autoregressive SMILES or diffusion-based 3D graph synthesis).

### 1.3 Strict Dependence on Precursor Mass & Adduct Annotations
- **Current Setup:** The mass-aware reranking multiplier ($S_{\text{mass}}$) relies on accurate estimation of the neutral monoisotopic mass from observed precursor $m/z$ and the metadata adduct label.
- **Vulnerability:**
  1. *Unknown or Misassigned Adducts:* In real untargeted LC-MS/MS experiments, adduct species are often unknown or misassigned (e.g., misinterpreting an in-source $[M - \text{H}_2\text{O} + \text{H}]^+$ fragment or an ammonium adduct $[M + \text{NH}_4]^+$ as $[M + \text{H}]^+$). An erroneous adduct shifts the calculated neutral mass by $17\text{–}18\text{ Da}$, completely extinguishing the Gaussian soft-mass score ($S_{\text{mass}} \approx 0.0$).
  2. *Low-Resolution Instruments:* While high-resolution Orbitrap or Q-TOF systems achieve $< 5\text{ ppm}$ mass accuracy, lower-resolution instruments (e.g., unit-resolution ion traps) exhibit mass errors $> 100\text{ ppm}$, violating the $\sigma = 20\text{ ppm}$ parameter.

### 1.4 Coarse Discretization of Fragment Mass Spectra
- **Current Setup:** Fragment ions are binned into 10,000 discrete intervals of $\Delta m/z = 0.1\text{ Da}$ over $[0, 1000\text{ Da}]$.
- **Limitation:** Modern high-resolution spectrometers routinely resolve isobaric fragment ions with identical nominal mass but distinct elemental compositions (e.g., $\text{CO}$ at $27.9944\text{ Da}$ vs. $\text{C}_2\text{H}_4$ at $28.0313\text{ Da}$, a difference of $0.0369\text{ Da}$). A bin width of $0.1\text{ Da}$ collapses these distinct elemental compositions into the same bin, discarding fine-grained formula information.

### 1.5 2D Topological Fingerprints vs. 3D Stereochemistry
- **Current Setup:** Molecules are encoded via 2048-bit ECFP4 Morgan fingerprints.
- **Limitation:** ECFP4 fingerprints capture 2D circular atom environments but do not encode 3D conformational dynamics, dihedral strain, or stereochemical configurations (enantiomers and diastereomers). While optical enantiomers often produce identical CID fragmentation, diastereomers can exhibit distinct fragment intensity ratios that 2D Morgan fingerprints cannot differentiate.

---

## 2. Completed Functionality vs. Proposed Future Work

| Component | Status | Implementation Details |
|---|:---:|---|
| **Sparse Spectrum Discretization (10,000 bins)** | **Completed** | `src/enveda/data/preprocessing.py:bin_spectrum_vector` |
| **Dual-Tower MLP Embedding Architecture** | **Completed** | `src/enveda/models/model.py:SpectrumMoleculeModel` |
| **Multi-Positive Contrastive Loss (MIL-NCE)** | **Completed** | `src/enveda/training/losses.py:multi_positive_contrastive_loss` |
| **Normalized Supervised Contrastive Loss (SupCon)** | **Completed** | `src/enveda/training/losses.py:normalized_supervised_contrastive_loss` |
| **Consensus Neutral Mass & Adduct Matching** | **Completed** | `src/enveda/evaluation/mass_filter.py:get_molecule_consensus_neutral_mass` |
| **Gaussian Soft Mass Reranking ($\sigma = 20\text{ ppm}$)** | **Completed** | `src/enveda/evaluation/mass_filter.py:compute_soft_mass_scores` |
| **Multi-Spectrum Molecule-Level Aggregation** | **Completed** | `src/enveda/evaluation/aggregation.py:aggregate_spectrum_scores` |
| **Chemistry-Aware Negative Sampling Pool** | **Completed** | `src/enveda/data/negative_sampler.py:ChemistryAwareNegativePool` |
| **Cosine Annealing Learning Rate Scheduler** | **Completed** | `src/enveda/training/schedulers.py:CosineWarmupScheduler` |
| **Embedding Diagnostics Tracker ($\text{Rank}_{\text{eff}}$)** | **Completed** | `src/enveda/training/diagnostics.py:EmbeddingDiagnosticsTracker` |
| *Graph Neural Network (GNN) Molecule Tower* | *Proposed* | Replace Morgan FP with directed message passing (D-MPNN) |
| *Continuous High-Resolution Peak Transformer* | *Proposed* | Replace fixed 0.1 Da bins with continuous 1D peak transformer |
| *Neural Adduct & Mass Hypothesis Classifier* | *Proposed* | Predict mixture distribution over candidate adduct states |
| *Cross-Attention Fragment-to-Substructure Reranker*| *Proposed* | Heavy second-stage reranker on top-50 candidates |

---

## 3. Detailed Proposed Future Directions

### 3.1 Directed Message Passing Neural Networks (D-MPNN) for Molecules
Replacing static binary Morgan fingerprints with a learned graph neural network (e.g. Chemprop or Uni-Mol) would allow the molecule tower to learn continuous atom and bond representations directly from molecular graphs. This enables the model to encode 3D spatial conformation and learn fragment-specific bond-cleavage propensities.

### 3.2 Continuous Peak Transformers for Mass Spectra
Instead of discretizing continuous spectra into 10,000 bins, modern attention-based architectures (e.g., SpecFormer or MassFormer) treat each spectrum as a variable-length sequence of continuous tokens:
$$t_i = \left[ \text{MLP}_{\text{mass}}(m/z_i) \,\|\, \text{MLP}_{\text{intensity}}(I_i) \right]$$
Using continuous sinusoidal positional encodings for $m/z$ values preserves the full sub-ppm mass resolution of Orbitrap spectrometers without expanding memory.

### 3.3 Neural Adduct & Charge State Classification
To eliminate vulnerability to missing or erroneous adduct annotations, a lightweight auxiliary classification head can be trained on the spectrum encoder representations to predict the likelihood distribution over the 15 common adduct types $P(\text{adduct} \mid \mathcal{S})$. The consensus neutral mass is then computed as a marginal expectation across plausible adduct hypotheses:
$$S_{\text{mass\_expected}}(c) = \sum_{a \in \mathcal{A}} P(a \mid \mathcal{S}) \cdot S_{\text{mass}}(c \mid a)$$
This provides robust soft-mass filtering even when experimental adduct metadata is absent.

### 3.4 Two-Stage Retrieval and Cross-Attention Reranking
In industrial search systems (e.g., web search and e-commerce), two-stage retrieval is standard:
1. **Stage 1 (Bi-Encoder):** Retrieve the top $K = 100$ candidate molecules in $< 10\text{ ms}$ using our dual-tower model and mass filter.
2. **Stage 2 (Cross-Encoder):** Pass the top 100 pairs through an expressive cross-attention network that explicitly models the interaction between individual fragment peaks and molecular substructure graphs, determining whether predicted *in silico* bond breakages match the observed peaks.

