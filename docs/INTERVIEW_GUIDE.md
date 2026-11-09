# Enveda CASMI 2026: Comprehensive Technical Interview Guide

**Document ID:** `docs/INTERVIEW_GUIDE.md`  
**Purpose:** Interview-oriented, multi-level explanations of the machine learning, cheminformatics, mass spectrometry, and systems engineering decisions in this repository.  

Each topic is answered across three rigorous tiers:
1. **Simple Explanation:** Intuitive, clear explanation accessible to any software or data engineer.
2. **Technically Accurate Explanation:** Chemically, physically, and mathematically rigorous formulation.
3. **How It Applies to This Project:** Concrete implementation details, parameter choices, file references, and measured empirical benchmarks.

---

## 1. What problem are we solving?

### Simple Explanation
When scientists discover an unknown chemical substance (like a potential new medicine in a plant), they shoot it through a machine that smashes it into fragments and weighs the pieces. We build an AI system that takes that list of fragment weights and searches a huge database of known molecules to figure out which chemical structure created the fragments.

### Technically Accurate Explanation
We solve the inverse problem of tandem mass spectrometry ($\text{MS/MS}$) molecular structure elucidation: mapping a continuous, variable-length set of fragment ion mass-to-charge ratios and intensities $\mathcal{S} = \{(m/z_i, I_i)\}_{i=1}^P$ back to its originating 2D molecular graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, evaluated as candidate ranking over an open chemical library of 277,566 candidate SMILES.

### How It Applies to This Project
In CASMI 2026, we are given 400 test molecules comprising 1,213 experimental spectra. For each query molecule, our pipeline outputs a ranked list of the top 25 candidate SMILES strings evaluated by Mean Reciprocal Rank at 25 (MRR@25).

---

## 2. Why is mass spectrometry useful?

### Simple Explanation
It can detect and measure chemicals at ultra-tiny quantities (picograms) faster and cheaper than any other technology, without having to isolate and crystallize large pure amounts of the chemical first.

### Technically Accurate Explanation
Mass spectrometry couples ultra-high analytical sensitivity (down to attomole detection limits) with high mass accuracy (< 5 ppm on Orbitrap/Q-TOF instruments), enabling high-throughput metabolomic, proteomic, and natural-product profiling of complex unpurified biological mixtures.

### How It Applies to This Project
Enveda Biosciences uses mass spectrometry to survey the "dark matter" of natural plant chemistry to identify novel bioactive drug scaffolds. Our model accelerates this discovery pipeline by automating structure identification from experimental spectra.

---

## 3. What is an MS/MS spectrum?

### Simple Explanation
In tandem mass spectrometry ($\text{MS/MS}$ or $\text{MS}^2$), the machine selects one intact molecule ion by its mass (the "precursor"), smashes it against inert gas molecules (like Nitrogen) so it shatters along its chemical bonds, and then records the mass and brightness of every fragment piece.

### Technically Accurate Explanation
Collision-Induced Dissociation (CID) imparts internal vibrational energy to the precursor ion $[M + \text{Adduct}]^{\pm}$, inducing unimolecular dissociation pathways along characteristic labile bonds. The resulting spectrum is a discrete distribution of fragment ion abundance as a function of mass-to-charge ratio $m/z$.

### How It Applies to This Project
Each record in `data/train.parquet` represents an MS/MS event with raw peak arrays (`mz_array`, `intensity_array`), precursor $m/z$, charge, collision energy, and adduct type. We process these fragment patterns to infer the parent molecule.

---

## 4. Why represent spectra as sparse vectors?

### Simple Explanation
Different experiments record different numbers of peaks at decimal mass positions. By dividing the mass scale from 0 to 1,000 into 10,000 tiny equal buckets (each 0.1 wide) and putting peak intensities into those buckets, every spectrum becomes a standard list of 10,000 numbers that a standard neural network can read.

### Technically Accurate Explanation
Mass spectra are inherently variable-length sets of Dirac-like delta functions on $\mathbb{R}^+$. Uniform discretization with bin width $\Delta m/z = 0.1\text{ Da}$ over the domain $[0, 1000\text{ Da}]$ projects the continuous spectrum into a fixed-dimensional Hilbert space $\mathbb{R}^{10000}$. Because most bins are empty, this representation is sparse, preserve fragment mass resolution, and enables high-throughput matrix multiplication on GPUs.

### How It Applies to This Project
Implemented in `src/enveda/data/preprocessing.py` via `bin_spectrum_vector`. Peaks are thresholded to relative intensity $\ge 0.01$, restricted to the top 128 peaks, binned into 10,000 dimensions, and L2-normalized.

---

## 5. Why use Morgan fingerprints?

### Simple Explanation
A computer cannot directly feed a 2D molecule drawing into basic matrix math. Morgan fingerprints walk around each atom in a molecule up to 2 steps away, take notes on the neighborhood chemical groups, and turn that into a binary barcode of 2,048 ones and zeros.

### Technically Accurate Explanation
Extended Connectivity Fingerprints (ECFP4 / Morgan Circular Fingerprints, radius 2) are topological representations derived via the Morgan algorithm. They iteratively hash circular atom environments (encompassing substructures up to 4 bonds across) into a fixed-length bit vector ($B = 2048$), capturing functional groups, ring assemblies, and heteroatom arrangements independent of atom numbering or 2D coordinate orientation.

### How It Applies to This Project
Generated using RDKit in `src/enveda/data/preprocessing.py` via `generate_morgan_fingerprint`. Every candidate SMILES is converted into a 2,048-dimensional float tensor that serves as input to the Molecule Encoder.

---

## 6. Why a dual-encoder (two-tower) architecture?

### Simple Explanation
If you used a single giant neural network that took both a spectrum and a candidate molecule together to compare them, searching 277,566 candidate molecules for 1 spectrum would require running that giant network 277,566 times (hours per spectrum!). With a two-tower model, you pre-calculate embeddings for all 277,566 candidates once. Then for a new spectrum, you encode it once and do a single matrix multiplication that finishes in milliseconds.

### Technically Accurate Explanation
A cross-attention or joint encoder has computational complexity $\mathcal{O}(Q \times C \times L^2)$, making candidate retrieval across $C = 277,566$ candidates intractable for inference. A dual-tower encoder decouples representation learning: candidate embeddings $z_{\text{mol}} \in \mathbb{S}^{D-1}$ are indexed offline, reducing online query evaluation to a single matrix multiplication $\mathcal{O}(Q \times D \times C)$ followed by top-$K$ selection on GPU.

### How It Applies to This Project
Implemented in `src/enveda/models/model.py` (`SpectrumEncoder` and `MoleculeEncoder`). Candidate embeddings for all 277,566 SMILES are precomputed and cached in `outputs/candidate_bank/` in under 3 minutes. Query retrieval takes under 1.5 seconds per 100 spectra.

---

## 7. How does contrastive learning work?

### Simple Explanation
It trains the AI like a multiple-choice test. The network looks at a spectrum and its true molecule (the positive match) mixed together with dozens of other unrelated molecules (the negative distractors). The loss function forces the model to increase the similarity score for the true match while pushing down the scores of all the distractors.

### Technically Accurate Explanation
Contrastive learning aligns multimodal representations by maximizing the mutual information lower bound between paired views (InfoNCE / MIL-NCE). It maps positive pairs $(x_{\text{spec}}, x_{\text{mol}}^+)$ to high cosine similarity while pushing negative pairs $(x_{\text{spec}}, x_{\text{mol}}^-)$ toward orthogonality or antipodality on the unit hypersphere:
$$\mathcal{L} = -\log \frac{\exp(\langle z_{\text{spec}}, z_{\text{mol}}^+ \rangle / \tau)}{\sum_j \exp(\langle z_{\text{spec}}, z_{\text{mol}}^{(j)} \rangle / \tau)}$$

### How It Applies to This Project
Implemented in `src/enveda/training/losses.py` with multi-positive extension and temperature scaling $\tau = 0.07$. When multiple spectra in a batch originate from the same molecule identity, they are masked as mutual positives rather than false negatives.

---

## 8. What are positive and negative pairs?

### Simple Explanation
- **Positive Pair:** An experimental mass spectrum paired with the actual chemical molecule that generated it.
- **Negative Pair:** A spectrum paired with any other chemical molecule that did not generate it.

### Technically Accurate Explanation
Under an empirical data distribution $p(\mathcal{S}, \mathcal{M})$, a positive pair is a joint sample $(s_i, m_i) \sim p(\mathcal{S}, \mathcal{M})$ sharing identical molecular identity (defined by 14-character InChIKey connectivity). A negative pair $(s_i, m_j)$ is an independent product-of-marginals sample where $m_j \ne m_i$.

### How It Applies to This Project
Positive masks are computed via `molecule_ids[:, None] == molecule_ids[None, :]` in `losses.py`. In Phase 6B Run C, negative pairs include both in-batch random negatives and external chemistry-aware negatives (mass-compatible and structural isomers).

---

## 9. Why use cosine similarity?

### Simple Explanation
Because raw vector lengths can vary, cosine similarity ignores the size/magnitude of vectors and only measures the angle between them. If two vectors point in the exact same direction, cosine similarity is 1.0; if they are perpendicular, it is 0.0.

### Technically Accurate Explanation
Because both tower outputs are L2-normalized ($\|z\|_2 = 1$), their Euclidean inner product is mathematically identical to the cosine of the angle between them:
$$\langle z_{\text{spec}}, z_{\text{mol}} \rangle = \cos(\theta)$$
Constraining embeddings to the compact unit hypersphere $\mathbb{S}^{D-1}$ prevents unbounded norm explosion, stabilizes contrastive gradient dynamics, and ensures scale-invariant similarity calculations.

### How It Applies to This Project
Both encoders end with `F.normalize(x, p=2, dim=-1)`. Similarity matrices are computed via standard linear algebra `spectrum_embeddings @ candidate_embeddings.T`.

---

## 10. What is precursor mass compatibility?

### Simple Explanation
A chemical molecule cannot produce fragment pieces that weigh more than the whole molecule itself, and the intact mass of the molecule must match what the mass spectrometer measured at the beginning of the experiment. If a candidate in the database has a different weight, it is physically impossible for it to be the true molecule.

### Technically Accurate Explanation
By the principle of conservation of mass, the monoisotopic neutral mass of the candidate molecule $M(c)$ must equal the neutral mass derived from the experimental precursor ion $m/z_{\text{prec}}$, charge state $z$, and adduct ion shift $\Delta m_{\text{adduct}}$:
$$M_{\text{neutral}} = m/z_{\text{prec}} \cdot |z| - \Delta m_{\text{adduct}}$$
Candidates whose exact mass violates this equality within instrument precision are chemically incompatible.

### How It Applies to This Project
Implemented in `src/enveda/evaluation/mass_filter.py`. Exact monoisotopic masses are calculated with RDKit (`ExactMolWt`) and stored in 32-bit floating point arrays.

---

## 11. What is ppm (parts-per-million)?

### Simple Explanation
It is a relative unit of measurement for precision. 1 ppm means an error of 1 milligram out of 1 kilogram, or 1 second in 11.5 days. In mass spectrometry, an instrument with 20 ppm error measuring a molecule of mass 500 Da might be off by at most $500 \times 20 / 10^6 = 0.01\text{ Da}$.

### Technically Accurate Explanation
Parts-per-million ($\text{ppm}$) is a dimensionless metric of relative measurement error:
$$\text{ppm} = \frac{|m_{\text{theoretical}} - m_{\text{experimental}}|}{m_{\text{theoretical}}} \times 10^6$$
Using relative ppm rather than absolute Daltons ($\text{Da}$) accounts for the mass-dependent scaling of mass spectrometry measurement uncertainty.

### How It Applies to This Project
Our candidate screening filter uses a $\pm 50\text{ ppm}$ tolerance window, and the soft mass scoring function uses a Gaussian scale parameter $\sigma = 20\text{ ppm}$.

---

## 12. Why use mass-aware scoring?

### Simple Explanation
Without mass scoring, the neural network might pick a molecule that looks structurally similar to the fragments but weighs twice as much as the real molecule. By multiplying the neural similarity by a mass compatibility score, we eliminate physically impossible molecules while letting the neural network choose among the true isomers that weigh the correct amount.

### Technically Accurate Explanation
Mass spectrometry fragment matching is an ill-posed inverse problem: many disparate substructures share identical topological fragments. Imposing a Gaussian soft mass prior $S_{\text{mass}} = \exp(-\frac{1}{2}(\Delta\text{ppm}/\sigma)^2)$ forms a joint maximum a posteriori (MAP) estimator:
$$P(\text{mol} \mid \text{spec}, m/z_{\text{prec}}) \propto P(\text{spec} \mid \text{mol}) \times P(m/z_{\text{prec}} \mid \text{mol})$$
Where $P(\text{spec} \mid \text{mol}) \approx \exp(\langle z_s, z_m \rangle)$ and $P(m/z_{\text{prec}} \mid \text{mol}) \approx S_{\text{mass}}$.

### How It Applies to This Project
Empirically validated in Phase 5: adding mass-aware scoring improved MRR@25 from **0.0006 to 0.2686** (+44,667% relative gain) and Recall@25 from **0.50% to 69.75%**.

---

## 13. What is MRR@25?

### Simple Explanation
Mean Reciprocal Rank truncated at 25: for each test molecule, if the true chemical is ranked #1, you get a perfect score of 1.0 ($1/1$). If it is ranked #2, you get 0.5 ($1/2$). If it is ranked #5, you get 0.2 ($1/5$). If it is ranked past 25, you get 0.0. The average of these scores across all test molecules is the MRR@25.

### Technically Accurate Explanation
The evaluation metric is defined as:
$$\text{MRR@25} = \frac{1}{M} \sum_{m=1}^M \begin{cases} \frac{1}{\text{rank}_m} & \text{if } \text{rank}_m \le 25 \\ 0 & \text{if } \text{rank}_m > 25 \end{cases}$$
Where $\text{rank}_m \in \{1, 2, \dots, |\mathcal{C}|\}$ is the 1-based index of the ground-truth molecule in the descending sorted candidate list.

### How It Applies to This Project
Implemented in `src/enveda/evaluation/metrics.py`. Our current champion achieves an official validation MRR@25 of **0.2686** across 400 held-out molecules.

---

## 14. Why is Recall@25 important?

### Simple Explanation
Chemists and automated robotic synthesis platforms can realistically review or test a shortlist of 25 candidate molecules in the lab. Recall@25 measures the percentage of test molecules where the true chemical made it onto that 25-candidate shortlist.

### Technically Accurate Explanation
$$\text{Recall@25} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}(\text{rank}_m \le 25)$$
Recall@25 defines the upper-bound coverage for any subsequent downstream stage (such as high-level DFT quantum simulation, robotic synthesis, or manual spectroscopic validation). If the true structure is not in the top 25, it is lost to the discovery pipeline.

### How It Applies to This Project
In Phase 6B Run C, our chemistry-aware training pushed Recall@25 to a project record **70.50%** (282 out of 400 validation molecules captured in top 25), approaching the theoretical 98.00% target survival ceiling.

---

## 15. What is embedding collapse?

### Simple Explanation
It is a failure mode where the neural network gets lazy and squishes all its output vectors into a narrow line or cone in the embedding space instead of using the entire sphere. When all vectors point in almost the same direction, the model loses the ability to distinguish different chemicals.

### Technically Accurate Explanation
Dimensional collapse occurs when the representation matrix $\mathbf{Z} \in \mathbb{R}^{N \times D}$ experiences rank deficiency, causing the singular values $\sigma_i$ to decay rapidly and collapsing representations onto a lower-dimensional subspace ($d_{\text{eff}} \ll D$). It is measured via **Effective Rank**:
$$\text{Rank}_{\text{eff}} = \exp\left(-\sum_{i=1}^D p_i \ln p_i\right), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$

### How It Applies to This Project
Diagnosed in Phase 6A: extended training for 20,000 steps caused Spectrum Encoder effective rank to drop from 78.1 down to 55.3. Tracking effective rank in `src/enveda/training/diagnostics.py` helped explain why Run C's MRR capped at 0.2525 despite record recall.

---

## 16. Why can training loss decrease while validation performance worsens?

### Simple Explanation
The model can learn "shortcuts" or overfit. In contrastive learning, the model might get really good at separating the specific training molecules from each other by memorizing trivial batch quirks, but it stops learning general chemical rules, so its ability to rank completely unseen molecules in the validation set begins to decay.

### Technically Accurate Explanation
Contrastive loss is an empirical surrogate objective over in-batch negatives. As training progresses, the gradient norm is dominated by trivial in-batch negative pairs ($T < 0.3$), causing the empirical cross-entropy loss to continue dropping as logit magnitudes sharpen. However, the model overfits to the training manifold and suffers feature suppression on fine-grained isomer differences, widening the validation generalization gap.

### How It Applies to This Project
Observed between Step 5,000 and Step 20,000: training loss fell from 1.95 to 1.62, while validation loss rose from 4.78 to 5.68, and validation MRR@25 declined from 0.2686 to 0.2604. This justified early stopping at Step 5,000.

---

## 17. Why use hard-negative mining?

### Simple Explanation
If you only test a student with ridiculously easy multiple-choice questions (like distinguishing a cat from an airplane), they never learn to distinguish a leopard from a jaguar. In chemistry, if the model only sees random negatives, it never learns how to tell two isomers apart. Hard negatives force the model to study the subtle differences between molecules with the exact same weight or similar structures.

### Technically Accurate Explanation
Random negative pairs in contrastive learning provide diminishing gradient signal:
$$\frac{\partial \mathcal{L}}{\partial z_s} \propto \sum_{j \in \text{neg}} \frac{\exp(z_s \cdot z_j / \tau)}{\sum_k \exp(z_s \cdot z_k / \tau)} z_j$$
When negatives are structurally dissimilar ($z_s \cdot z_j \ll 0$), their softmax probability approaches zero, producing vanishing gradients. Hard-negative mining selects negative candidates satisfying $z_s \cdot z_j \approx z_s \cdot z_p$ (e.g. mass-compatible isomers within 50 ppm and Morgan Tanimoto $T \ge 0.3$), ensuring informative, high-magnitude gradient updates.

### How It Applies to This Project
Implemented in `src/enveda/data/negative_sampler.py` (`ChemistryAwareNegativePool`). Audited across 1,000 anchors, the sampler provided 44.8% mass-compatible and 27.6% structural hard negatives, driving Recall@25 to an all-time project high of 70.50%.

---

## 18. What did each experiment teach us?

### Simple Explanation
- **Baseline (Phase 1–4):** Neural embeddings alone cannot guess molecules without mass knowledge (0.05% accuracy).
- **Mass Scoring (Phase 5):** Combining neural similarity with physics-based precursor mass scoring is transformative (+44,600% gain, MRR@25 = 0.2686).
- **Training Duration (Phase 6A):** Training too long at constant learning rate degrades performance due to negative starvation and dimensional collapse.
- **Cosine Schedule (Phase 6B Run A):** Annealing the learning rate stabilizes updates but doesn't solve isomer confusion alone.
- **Chemistry Negatives (Phase 6B Run C):** Hard negatives significantly boost candidate coverage (Recall@25 to 70.50%), but un-normalized MIL-NCE contracts the embedding space, capping top-1 MRR.
- **Final Decision:** Preserve the proven, isotropic Step 5,000 champion (MRR@25 = 0.2686).

### Technically Accurate Explanation
- **Ablation Study:** Demonstrated that neural cosine similarity and Gaussian soft mass priors are orthogonal and mutually necessary signals ($S = S_{\text{neural}} \times S_{\text{mass}}$).
- **Optimization Dynamics:** Confirmed that multi-positive MIL-NCE without $\frac{1}{|P|}$ normalization introduces multiplicity bias when scaled with hard negative pools, motivating Khosla Normalized SupCon for future iterations.
- **Empirical Rigor:** Enforced strict, reproducible evaluation over an exact 277,566 candidate universe on zero-leakage InChIKey-split validation sets.
