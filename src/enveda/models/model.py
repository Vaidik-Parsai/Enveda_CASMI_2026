"""Two-tower neural network architecture for spectrum-to-molecule retrieval.

Strictly preserves parameter names, tensor dimensions, and layer hierarchies
for 100% backwards compatibility with existing checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    """Encode binned spectrum data and experimental metadata."""

    def __init__(
        self,
        spectrum_dim: int = 10000,
        metadata_dim: int = 130,
        embedding_dim: int = 128,
        dropout_1: float = 0.2,
        dropout_2: float = 0.1,
    ):
        super().__init__()
        input_dim = spectrum_dim + metadata_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, spectrum: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Forward pass concatenating spectrum and tabular metadata.
        
        Args:
            spectrum: Tensor of shape (batch_size, spectrum_dim).
            metadata: Tensor of shape (batch_size, metadata_dim).
            
        Returns:
            L2-normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        x = torch.cat([spectrum, metadata], dim=1)
        x = self.network(x)
        return F.normalize(x, p=2, dim=1)


class MoleculeEncoder(nn.Module):
    """Encode Morgan molecular fingerprints."""

    def __init__(
        self,
        fingerprint_dim: int = 2048,
        embedding_dim: int = 128,
        dropout_1: float = 0.2,
        dropout_2: float = 0.1,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(fingerprint_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting fingerprint to latent space.
        
        Args:
            fingerprint: Tensor of shape (batch_size, fingerprint_dim).
            
        Returns:
            L2-normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        x = self.network(fingerprint)
        return F.normalize(x, p=2, dim=1)


class SpectrumMoleculeModel(nn.Module):
    """Two-tower retrieval model combining SpectrumEncoder and MoleculeEncoder."""

    def __init__(
        self,
        spectrum_dim: int = 10000,
        metadata_dim: int = 130,
        fingerprint_dim: int = 2048,
        embedding_dim: int = 128,
        dropout_1: float = 0.2,
        dropout_2: float = 0.1,
    ):
        super().__init__()
        self.spectrum_encoder = SpectrumEncoder(
            spectrum_dim=spectrum_dim,
            metadata_dim=metadata_dim,
            embedding_dim=embedding_dim,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
        )

        self.molecule_encoder = MoleculeEncoder(
            fingerprint_dim=fingerprint_dim,
            embedding_dim=embedding_dim,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
        )

    def encode_spectrum(self, spectrum: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Produce L2-normalized spectrum embedding."""
        return self.spectrum_encoder(spectrum, metadata)

    def encode_molecule(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Produce L2-normalized molecule embedding."""
        return self.molecule_encoder(fingerprint)

    def forward(
        self,
        spectrum: torch.Tensor,
        metadata: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        """Run both towers simultaneously."""
        spectrum_embedding = self.encode_spectrum(spectrum, metadata)
        molecule_embedding = self.encode_molecule(fingerprint)
        return spectrum_embedding, molecule_embedding
