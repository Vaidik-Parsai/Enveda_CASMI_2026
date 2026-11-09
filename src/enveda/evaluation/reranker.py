"""Lightweight candidate reranker combining neural retrieval, mass accuracy, and cross-spectra agreement."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from enveda.evaluation.mass_filter import (
    calculate_exact_monoisotopic_mass,
    compute_soft_mass_scores,
)


class CandidateReranker:
    """Lightweight scoring model for reranking top candidate molecules."""

    def __init__(
        self,
        weight_neural: float = 1.0,
        weight_mass_ppm: float = 2.0,
        sigma_ppm: float = 25.0,
        weight_consistency: float = 0.5,
    ):
        self.weight_neural = weight_neural
        self.weight_mass_ppm = weight_mass_ppm
        self.sigma_ppm = sigma_ppm
        self.weight_consistency = weight_consistency

    def score_candidates(
        self,
        candidate_indices: List[int],
        candidate_masses: np.ndarray,
        expected_neutral_mass: float,
        spectrum_sims: np.ndarray,  # shape (num_spectra, len(candidate_indices))
    ) -> np.ndarray:
        """Compute composite reranking score for retrieved candidate pool.
        
        Args:
            candidate_indices: List of candidate indices in the candidate bank.
            candidate_masses: Array of exact masses for all candidates in the bank.
            expected_neutral_mass: Consensus expected neutral mass of query molecule.
            spectrum_sims: 2D array of similarities (num_spectra, pool_size).
            
        Returns:
            1D array of composite scores of length len(candidate_indices).
        """
        pool_masses = candidate_masses[candidate_indices]

        # 1. Neural Score (mean similarity across spectra)
        mean_neural = np.mean(spectrum_sims, axis=0)  # shape (pool_size,)

        # 2. Mass Accuracy Score: S_mass = exp(-0.5 * (ppm_diff / sigma_ppm)^2)
        delta_mass = np.abs(pool_masses - expected_neutral_mass)
        ppm_diff = (delta_mass / expected_neutral_mass) * 1e6
        mass_score = np.exp(-0.5 * (ppm_diff / self.sigma_ppm) ** 2)

        # 3. Cross-Spectrum Consistency (low variance across spectra gives bonus)
        if spectrum_sims.shape[0] > 1:
            std_sim = np.std(spectrum_sims, axis=0)
            consistency_score = 1.0 / (1.0 + std_sim)
        else:
            consistency_score = np.ones_like(mean_neural)

        # Composite score
        # Neural similarity (shifted to [0, 1]) multiplied by mass compatibility and consistency
        composite_score = (
            (mean_neural + 1.0)
            * (mass_score ** self.weight_mass_ppm)
            * (consistency_score ** self.weight_consistency)
        )
        return composite_score

    def rerank(
        self,
        candidate_indices: List[int],
        candidate_masses: np.ndarray,
        expected_neutral_mass: float,
        spectrum_sims: np.ndarray,
        top_k: int = 25,
    ) -> Tuple[List[int], List[float]]:
        """Rerank candidate pool and return top-k candidate indices with scores."""
        scores = self.score_candidates(
            candidate_indices=candidate_indices,
            candidate_masses=candidate_masses,
            expected_neutral_mass=expected_neutral_mass,
            spectrum_sims=spectrum_sims,
        )
        sorted_order = np.argsort(-scores)[:top_k]
        reranked_indices = [candidate_indices[idx] for idx in sorted_order]
        reranked_scores = [float(scores[idx]) for idx in sorted_order]
        return reranked_indices, reranked_scores

