"""Embedding diagnostics and dimensional collapse tracking."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class EmbeddingDiagnosticsTracker:
    """Tracks geometric representation metrics across training steps.
    
    Monitors:
    - Effective rank (Roy & Vetterli, 2007) of spectrum and molecule embeddings.
    - Mean and standard deviation of positive vs negative cosine similarities.
    - Validation positive alignment.
    - Mean embedding variance and off-diagonal covariance (VICReg/Barlow Twins style).
    - Generalization gap (validation loss minus training loss).
    """

    @staticmethod
    def compute_effective_rank(embeddings: torch.Tensor) -> float:
        """Compute the Shannon entropy effective rank of an embedding matrix."""
        if embeddings.ndim != 2 or embeddings.shape[0] < 2:
            return 1.0

        # Center embeddings
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        p = s / (s.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return float(torch.exp(entropy).item())

    @staticmethod
    def compute_covariance_metrics(embeddings: torch.Tensor) -> Tuple[float, float]:
        """Compute mean feature variance and mean off-diagonal covariance.
        
        High variance and near-zero off-diagonal covariance indicate non-collapsed,
        orthogonal feature representations.
        
        Returns:
            Tuple of (mean_variance, mean_abs_off_diagonal_covariance).
        """
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (embeddings.shape[0] - 1 + 1e-8)
        diag = torch.diag(cov)
        mean_var = float(diag.mean().item())

        # Off-diagonal covariance
        off_diag = cov - torch.diag(diag)
        d = embeddings.shape[1]
        mean_off_diag = float((off_diag.abs().sum() / (d * (d - 1) + 1e-8)).item())
        return mean_var, mean_off_diag

    @staticmethod
    def evaluate_representation_geometry(
        spectrum_embeddings: torch.Tensor,
        molecule_embeddings: torch.Tensor,
        molecule_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute a full geometric snapshot on a validation batch or split."""
        spec_rank = EmbeddingDiagnosticsTracker.compute_effective_rank(spectrum_embeddings)
        mol_rank = EmbeddingDiagnosticsTracker.compute_effective_rank(molecule_embeddings)

        spec_var, spec_off_cov = EmbeddingDiagnosticsTracker.compute_covariance_metrics(spectrum_embeddings)
        mol_var, mol_off_cov = EmbeddingDiagnosticsTracker.compute_covariance_metrics(molecule_embeddings)

        # Cosine similarity matrix (assumes L2 normalized embeddings)
        sim_matrix = spectrum_embeddings @ molecule_embeddings.T
        pos_mask = (molecule_ids[:, None] == molecule_ids[None, :])
        neg_mask = ~pos_mask

        pos_sims = sim_matrix[pos_mask]
        neg_sims = sim_matrix[neg_mask]

        pos_mean = float(pos_sims.mean().item()) if len(pos_sims) > 0 else 0.0
        pos_std = float(pos_sims.std().item()) if len(pos_sims) > 1 else 0.0
        neg_mean = float(neg_sims.mean().item()) if len(neg_sims) > 0 else 0.0
        neg_std = float(neg_sims.std().item()) if len(neg_sims) > 1 else 0.0
        margin = pos_mean - neg_mean

        return {
            "spectrum_effective_rank": spec_rank,
            "molecule_effective_rank": mol_rank,
            "spectrum_mean_variance": spec_var,
            "spectrum_off_diagonal_cov": spec_off_cov,
            "molecule_mean_variance": mol_var,
            "molecule_off_diagonal_cov": mol_off_cov,
            "positive_cosine_mean": pos_mean,
            "positive_cosine_std": pos_std,
            "negative_cosine_mean": neg_mean,
            "negative_cosine_std": neg_std,
            "cosine_margin": margin,
        }

