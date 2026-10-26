"""Contrastive loss functions for spectrum-molecule retrieval."""

import torch


def multi_positive_contrastive_loss(
    spectrum_embeddings: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    molecule_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Train spectrum and molecule embeddings to match in shared latent space.
    
    Multiple spectra belonging to the same molecule identity are treated as valid positives.
    
    Args:
        spectrum_embeddings: Tensor of shape (batch_size, embedding_dim).
        molecule_embeddings: Tensor of shape (batch_size, embedding_dim).
        molecule_ids: Integer identity for each molecule, shape (batch_size,).
        temperature: Temperature scaling factor for logits.
        
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

    # Mask for positive matches across the batch
    positive_mask = molecule_ids[:, None] == molecule_ids[None, :]

    lowest_value = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, lowest_value)

    # Spectrum -> Molecule direction
    loss_s2m = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)

    # Molecule -> Spectrum direction
    loss_m2s = torch.logsumexp(logits.T, dim=1) - torch.logsumexp(positive_logits.T, dim=1)

    # Symmetric average
    loss = 0.5 * (loss_s2m.mean() + loss_m2s.mean())
    return loss
