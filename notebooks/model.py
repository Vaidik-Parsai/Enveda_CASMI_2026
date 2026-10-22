import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    """Encode spectrum data and experimental metadata."""

    def __init__(
        self,
        spectrum_dim=10000,
        metadata_dim=130,
        embedding_dim=128,
    ):
        super().__init__()

        input_dim = spectrum_dim + metadata_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, embedding_dim),
        )

    def forward(self, spectrum, metadata):
        x = torch.cat([spectrum, metadata], dim=1)
        x = self.network(x)

        return F.normalize(x, p=2, dim=1)


class MoleculeEncoder(nn.Module):
    """Encode Morgan molecular fingerprints."""

    def __init__(
        self,
        fingerprint_dim=2048,
        embedding_dim=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, embedding_dim),
        )

    def forward(self, fingerprint):
        x = self.network(fingerprint)

        return F.normalize(x, p=2, dim=1)


class SpectrumMoleculeModel(nn.Module):
    """Two-tower model for spectrum-to-molecule retrieval."""

    def __init__(
        self,
        spectrum_dim=10000,
        metadata_dim=130,
        fingerprint_dim=2048,
        embedding_dim=128,
    ):
        super().__init__()

        self.spectrum_encoder = SpectrumEncoder(
            spectrum_dim=spectrum_dim,
            metadata_dim=metadata_dim,
            embedding_dim=embedding_dim,
        )

        self.molecule_encoder = MoleculeEncoder(
            fingerprint_dim=fingerprint_dim,
            embedding_dim=embedding_dim,
        )

    def encode_spectrum(self, spectrum, metadata):
        return self.spectrum_encoder(spectrum, metadata)

    def encode_molecule(self, fingerprint):
        return self.molecule_encoder(fingerprint)

    def forward(self, spectrum, metadata, fingerprint):
        spectrum_embedding = self.encode_spectrum(
            spectrum, metadata
        )
        molecule_embedding = self.encode_molecule(fingerprint)

        return spectrum_embedding, molecule_embedding

    
def multi_positive_contrastive_loss(
    spectrum_embeddings,
    molecule_embeddings,
    molecule_ids,
    temperature=0.07,
):
    """
    Train spectrum and molecule embeddings to match.

    Multiple spectra belonging to the same molecule
    are treated as valid positive matches.

    Args:
        spectrum_embeddings: Tensor of shape (batch_size, embedding_dim)
        molecule_embeddings: Tensor of shape (batch_size, embedding_dim)
        molecule_ids: Integer identity for each molecule, shape (batch_size,)
        temperature: Controls the sharpness of similarity scores.

    Returns:
        Scalar contrastive loss.
    """

    if spectrum_embeddings.ndim != 2:
        raise ValueError("Spectrum embeddings must be 2D.")

    if molecule_embeddings.shape != spectrum_embeddings.shape:
        raise ValueError(
            "Spectrum and molecule embeddings must have matching shapes."
        )

    if molecule_ids.ndim != 1:
        raise ValueError("molecule_ids must be a 1D tensor.")

    if len(molecule_ids) != len(spectrum_embeddings):
        raise ValueError("Batch sizes do not match.")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Cosine similarities, since the embeddings are normalized.
    # Shape: (batch_size, batch_size)
    logits = (
        spectrum_embeddings @ molecule_embeddings.T
    ) / temperature

    # True wherever two examples represent the same molecule.
    # Shape: (batch_size, batch_size)
    positive_mask = (
        molecule_ids[:, None] == molecule_ids[None, :]
    )

    # Spectrum -> molecule:
    # The numerator includes ALL matching molecules.
    lowest_value = torch.finfo(logits.dtype).min

    positive_logits = logits.masked_fill(
        ~positive_mask, lowest_value
    )

    loss_spectrum_to_molecule = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    )

    # Molecule -> spectrum:
    # Apply the same objective in the reverse direction.
    loss_molecule_to_spectrum = (
        torch.logsumexp(logits.T, dim=1)
        - torch.logsumexp(positive_logits.T, dim=1)
    )

    # Symmetric contrastive loss.
    loss = 0.5 * (
        loss_spectrum_to_molecule.mean()
        + loss_molecule_to_spectrum.mean()
    )

    return loss
