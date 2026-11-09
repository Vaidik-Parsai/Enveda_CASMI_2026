"""Contrastive loss functions for spectrum-molecule retrieval with hard-negative mining."""

from typing import Optional
import torch


def compute_batch_tanimoto(fingerprints: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Morgan Tanimoto similarity matrix for a batch of binary fingerprints on GPU.
    
    Args:
        fingerprints: Tensor of shape (batch_size, fingerprint_dim).
        
    Returns:
        Tensor of shape (batch_size, batch_size) with Tanimoto similarity values in [0, 1].
    """
    fp = (fingerprints > 0.5).float()
    intersection = fp @ fp.T
    sum_bits = fp.sum(dim=1)
    union = sum_bits[:, None] + sum_bits[None, :] - intersection
    union = torch.clamp(union, min=1.0)
    return intersection / union


def multi_positive_contrastive_loss(
    spectrum_embeddings: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    molecule_ids: torch.Tensor,
    temperature: float = 0.07,
    fingerprints: Optional[torch.Tensor] = None,
    hard_negative_weight: float = 1.0,
    hard_tanimoto_threshold: float = 0.4,
    negative_molecule_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Train spectrum and molecule embeddings with multi-positive and optional external negative support.
    
    Args:
        spectrum_embeddings: Tensor of shape (batch_size, embedding_dim).
        molecule_embeddings: Tensor of shape (batch_size, embedding_dim).
        molecule_ids: Integer identity for each molecule, shape (batch_size,).
        temperature: Temperature scaling factor for logits.
        fingerprints: Optional batch fingerprints (batch_size, fp_dim) for Morgan Tanimoto calculation.
        hard_negative_weight: Multiplier weight (> 1.0) applied to hard structural negatives.
        hard_tanimoto_threshold: Minimum Tanimoto similarity to qualify as a hard negative (default 0.4).
        negative_molecule_embeddings: Optional Tensor of external negative embeddings (num_negatives, embedding_dim).
        
    Returns:
        Scalar symmetric contrastive loss.
    """
    if spectrum_embeddings.ndim != 2:
        raise ValueError("Spectrum embeddings must be 2D.")

    if molecule_embeddings.shape != spectrum_embeddings.shape:
        raise ValueError("Spectrum and molecule embeddings must have matching shapes.")

    if molecule_ids.ndim != 1:
        raise ValueError("molecule_ids must be a 1D tensor.")

    if len(molecule_ids) != len(spectrum_embeddings):
        raise ValueError("Batch sizes do not match.")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Cosine similarities (since embeddings are L2 normalized)
    logits = (spectrum_embeddings @ molecule_embeddings.T) / temperature

    # Mask for positive matches across the batch (strictly by molecule identity)
    positive_mask = molecule_ids[:, None] == molecule_ids[None, :]

    # Hard Negative Mining: upweight negatives with high Morgan Tanimoto similarity
    logits_mod = logits
    if fingerprints is not None and hard_negative_weight > 1.0:
        tanimoto = compute_batch_tanimoto(fingerprints)
        # Never treat matching identities as negatives:
        hard_neg_mask = (~positive_mask) & (tanimoto >= hard_tanimoto_threshold)
        log_boost = torch.log(torch.tensor(float(hard_negative_weight), device=logits.device, dtype=logits.dtype))
        logits_mod = logits + log_boost * hard_neg_mask.to(logits.dtype)

    lowest_value = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, lowest_value)

    # If external sampled negatives are provided, append them to the denominator
    if negative_molecule_embeddings is not None and len(negative_molecule_embeddings) > 0:
        neg_logits = (spectrum_embeddings @ negative_molecule_embeddings.T) / temperature
        all_logits = torch.cat([logits_mod, neg_logits], dim=1)
    else:
        all_logits = logits_mod

    # Spectrum -> Molecule direction
    loss_s2m = torch.logsumexp(all_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)

    # Molecule -> Spectrum direction
    loss_m2s = torch.logsumexp(logits_mod.T, dim=1) - torch.logsumexp(positive_logits.T, dim=1)

    # Symmetric average
    loss = 0.5 * (loss_s2m.mean() + loss_m2s.mean())
    return loss


def normalized_supervised_contrastive_loss(
    spectrum_embeddings: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    molecule_ids: torch.Tensor,
    temperature: float = 0.07,
    fingerprints: Optional[torch.Tensor] = None,
    hard_negative_weight: float = 1.0,
    hard_tanimoto_threshold: float = 0.3,
    negative_molecule_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalized Supervised Contrastive Loss (SupCon) with multi-positive and external negative support."""
    if spectrum_embeddings.ndim != 2:
        raise ValueError("Spectrum embeddings must be 2D.")

    if molecule_embeddings.shape != spectrum_embeddings.shape:
        raise ValueError("Spectrum and molecule embeddings must have matching shapes.")

    if molecule_ids.ndim != 1 or len(molecule_ids) != len(spectrum_embeddings):
        raise ValueError("molecule_ids must be a 1D tensor matching batch size.")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Cosine similarities scaled by temperature
    logits = (spectrum_embeddings @ molecule_embeddings.T) / temperature

    # Positive mask: identical molecule identity
    positive_mask = (molecule_ids[:, None] == molecule_ids[None, :])
    pos_counts_s2m = positive_mask.sum(dim=1).clamp(min=1).float()  # |P(i)|
    pos_counts_m2s = positive_mask.sum(dim=0).clamp(min=1).float()  # |P(k)|

    # Hard-negative weighting
    logits_mod = logits
    if fingerprints is not None and hard_negative_weight > 1.0:
        tanimoto = compute_batch_tanimoto(fingerprints)
        # Strictly exclude all positives from hard-negative penalty
        hard_neg_mask = (~positive_mask) & (tanimoto >= hard_tanimoto_threshold)
        log_boost = torch.log(torch.tensor(float(hard_negative_weight), device=logits.device, dtype=logits.dtype))
        logits_mod = logits + log_boost * hard_neg_mask.to(logits.dtype)

    # If external sampled negatives are provided, append them to the denominator
    if negative_molecule_embeddings is not None and len(negative_molecule_embeddings) > 0:
        neg_logits = (spectrum_embeddings @ negative_molecule_embeddings.T) / temperature
        all_logits = torch.cat([logits_mod, neg_logits], dim=1)
    else:
        all_logits = logits_mod

    # 1. Spectrum -> Molecule direction
    denom_s2m = torch.logsumexp(all_logits, dim=1)
    pos_logits_s2m = torch.where(positive_mask, logits, torch.zeros_like(logits))
    mean_pos_s2m = pos_logits_s2m.sum(dim=1) / pos_counts_s2m
    loss_s2m = denom_s2m - mean_pos_s2m

    # 2. Molecule -> Spectrum direction
    denom_m2s = torch.logsumexp(logits_mod.T, dim=1)
    pos_logits_m2s = torch.where(positive_mask.T, logits.T, torch.zeros_like(logits.T))
    mean_pos_m2s = pos_logits_m2s.sum(dim=1) / pos_counts_m2s
    loss_m2s = denom_m2s - mean_pos_m2s

    # Symmetric average
    return 0.5 * (loss_s2m.mean() + loss_m2s.mean())

