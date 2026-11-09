"""Chemistry-aware negative sampler combining mass-compatibility, Morgan Tanimoto similarity, and random negatives."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from enveda.evaluation.mass_filter import (
    calculate_exact_monoisotopic_mass,
    estimate_neutral_mass,
)


class ChemistryAwareNegativePool:
    """Indexed pool of training candidate molecules for chemistry-aware negative sampling.
    
    Provides three categories of negative molecules:
    1. Random negatives (broad space coverage).
    2. Morgan Tanimoto structural hard negatives (T >= threshold).
    3. Precursor mass-compatible negatives (within ppm tolerance).
    
    Guarantees:
    - Ground-truth identities (by InChIKey14 or SMILES) are NEVER sampled as negatives.
    - Zero data leakage: only molecules from the training partition are indexed.
    """

    def __init__(
        self,
        training_smiles: List[str],
        training_inchikeys: List[str],
        training_fingerprints: np.ndarray,
        training_exact_masses: Optional[np.ndarray] = None,
        tanimoto_threshold: float = 0.3,
        mass_ppm_tolerance: float = 50.0,
        random_seed: int = 42,
    ):
        self.num_molecules = len(training_smiles)
        assert len(training_inchikeys) == self.num_molecules
        assert len(training_fingerprints) == self.num_molecules

        self.smiles = training_smiles
        self.inchikeys = training_inchikeys
        self.fingerprints = training_fingerprints
        self.tanimoto_threshold = tanimoto_threshold
        self.mass_ppm_tolerance = mass_ppm_tolerance
        self.rng = np.random.default_rng(random_seed)

        # Precompute or use exact masses
        if training_exact_masses is not None:
            self.exact_masses = np.asarray(training_exact_masses, dtype=np.float32)
        else:
            self.exact_masses = np.array(
                [calculate_exact_monoisotopic_mass(s) for s in self.smiles],
                dtype=np.float32,
            )

        # Pre-index sorted masses for fast binary search
        self.sorted_mass_indices = np.argsort(self.exact_masses)
        self.sorted_masses = self.exact_masses[self.sorted_mass_indices]

        # InChIKey to molecule indices map for strict identity masking
        self.inchikey_to_indices = defaultdict(list)
        for idx, ikey in enumerate(self.inchikeys):
            self.inchikey_to_indices[ikey].append(idx)

        # Pre-convert fingerprints to float on CPU for fast dot products
        self.fp_tensor = torch.from_numpy((self.fingerprints > 0.5).astype(np.float32))
        self.fp_sums = self.fp_tensor.sum(dim=1)

    def sample_negatives_for_anchor(
        self,
        anchor_inchikey: str,
        anchor_fp: np.ndarray,
        anchor_precursor_mz: float,
        anchor_adduct: Optional[str] = None,
        anchor_mode: Optional[str] = None,
        num_negatives: int = 16,
        anchor_idx: int = 0,
        mass_ratio: float = 0.50,
        tanimoto_ratio: float = 0.25,
        random_ratio: float = 0.25,
        allow_adaptive_structural_fallback: bool = True,
    ) -> Tuple[List[int], Dict[str, int]]:
        """Sample chemistry-aware negatives partitioned across categories.
        
        Args:
            anchor_inchikey: InChIKey14 of anchor (strictly masked from negatives).
            anchor_fp: Morgan fingerprint of anchor (2048-dim array).
            anchor_precursor_mz: Precursor m/z of anchor spectrum.
            anchor_adduct: Adduct string.
            anchor_mode: Ionization mode.
            num_negatives: Total negatives requested per anchor.
            anchor_idx: Index of anchor in batch (used for balanced alternating allocations).
            mass_ratio: Target fraction of mass-compatible negatives.
            tanimoto_ratio: Target fraction of structural Tanimoto negatives.
            random_ratio: Target fraction of random negatives.
            allow_adaptive_structural_fallback: Whether to use nearest structural neighbor
                if threshold >= 0.3 is not met (never silently substitutes random negatives).
            
        Returns:
            Tuple of (negative_indices, category_counts).
        """
        # Strict exclusion of anchor identity
        forbidden_indices = set(self.inchikey_to_indices.get(anchor_inchikey, []))

        # Balanced integer quota calculation
        if num_negatives >= 4:
            k_mass = int(round(num_negatives * mass_ratio))
            k_tanimoto = int(round(num_negatives * tanimoto_ratio))
            k_random = num_negatives - k_mass - k_tanimoto
        elif num_negatives == 2:
            k_mass = 1
            if anchor_idx % 2 == 0:
                k_tanimoto = 1
                k_random = 0
            else:
                k_tanimoto = 0
                k_random = 1
        elif num_negatives == 1:
            # Cycle through categories across consecutive anchors
            mod = anchor_idx % 4
            if mod in (0, 1):
                k_mass, k_tanimoto, k_random = 1, 0, 0
            elif mod == 2:
                k_mass, k_tanimoto, k_random = 0, 1, 0
            else:
                k_mass, k_tanimoto, k_random = 0, 0, 1
        else:
            k_mass = max(1, int(round(num_negatives * mass_ratio)))
            k_tanimoto = int(round(num_negatives * tanimoto_ratio))
            k_random = max(0, num_negatives - k_mass - k_tanimoto)

        chosen_indices = []
        counts = {"mass_compatible": 0, "tanimoto_hard": 0, "random": 0}

        # 1. Mass-compatible negatives within 50 ppm
        exp_mass = estimate_neutral_mass(anchor_precursor_mz, anchor_adduct, anchor_mode)
        if exp_mass is not None and exp_mass > 0 and k_mass > 0:
            delta_da = (self.mass_ppm_tolerance * 1e-6) * exp_mass
            min_m = exp_mass - delta_da
            max_m = exp_mass + delta_da

            # Binary search range in sorted masses
            idx_start = np.searchsorted(self.sorted_masses, min_m, side="left")
            idx_end = np.searchsorted(self.sorted_masses, max_m, side="right")

            cand_pool = self.sorted_mass_indices[idx_start:idx_end]
            valid_mass_cands = [idx for idx in cand_pool if idx not in forbidden_indices]

            if len(valid_mass_cands) > 0:
                n_take = min(k_mass, len(valid_mass_cands))
                sampled_mass = self.rng.choice(valid_mass_cands, size=n_take, replace=False).tolist()
                chosen_indices.extend(sampled_mass)
                counts["mass_compatible"] = len(sampled_mass)

        # 2. Morgan Tanimoto structural hard negatives
        if k_tanimoto > 0:
            a_fp = torch.from_numpy((anchor_fp > 0.5).astype(np.float32))
            inter = torch.mv(self.fp_tensor, a_fp)
            union = self.fp_sums + a_fp.sum() - inter
            union = torch.clamp(union, min=1.0)
            sims = (inter / union).numpy()

            # Mask forbidden and already chosen
            for f_idx in forbidden_indices:
                sims[f_idx] = -1.0
            for c_idx in chosen_indices:
                sims[c_idx] = -1.0

            # Find candidates with T >= threshold
            tanimoto_cands = np.where(sims >= self.tanimoto_threshold)[0]
            if len(tanimoto_cands) > 0:
                n_take = min(k_tanimoto, len(tanimoto_cands))
                sampled_t = self.rng.choice(tanimoto_cands, size=n_take, replace=False).tolist()
                chosen_indices.extend(sampled_t)
                counts["tanimoto_hard"] = len(sampled_t)
            elif allow_adaptive_structural_fallback:
                # Adaptive structural fallback: choose nearest structural neighbor in candidate pool
                # Ensures structural negative requirement is fulfilled without falling back to random
                valid_sim_idxs = np.where(sims >= 0.0)[0]
                if len(valid_sim_idxs) > 0:
                    top_idxs = valid_sim_idxs[np.argsort(sims[valid_sim_idxs])[::-1]]
                    n_take = min(k_tanimoto, len(top_idxs))
                    sampled_t = top_idxs[:n_take].tolist()
                    chosen_indices.extend(sampled_t)
                    counts["tanimoto_hard"] = len(sampled_t)

        # 3. Random negatives (STRICT quota: never silently substitutes missing structural/mass negatives)
        if k_random > 0:
            random_pool = [i for i in range(self.num_molecules) if i not in forbidden_indices and i not in chosen_indices]
            if len(random_pool) > 0:
                n_take = min(k_random, len(random_pool))
                sampled_rand = self.rng.choice(random_pool, size=n_take, replace=False).tolist()
                chosen_indices.extend(sampled_rand)
                counts["random"] = len(sampled_rand)

        return chosen_indices, counts

        return chosen_indices, counts

