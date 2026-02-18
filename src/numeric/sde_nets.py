"""
Latent SDE coefficient networks: DriftNet and DiffusionNet.

DriftNet predicts the latent drift b_z(z) ∈ R^d.
DiffusionNet predicts the latent diffusion σ_z(z) as a lower-triangular d×d matrix,
so that the latent covariance is Σ_z = σ_z σ_z^T (always PSD).
"""
import torch
import torch.nn as nn
from torch import Tensor

from .ffnn import FeedForwardNeuralNet


class DriftNet(nn.Module):
    """MLP: R^d -> R^d. Predicts latent drift b_z(z)."""

    def __init__(self, latent_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        neurons = [latent_dim] + hidden_dims + [latent_dim]
        activations = [nn.Tanh()] * len(hidden_dims) + [None]
        self.net = FeedForwardNeuralNet(neurons, activations)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, d)
        Returns:
            b_z: (B, d) latent drift vector
        """
        return self.net(z)


class DiffusionNet(nn.Module):
    """MLP: R^d -> lower-triangular d×d. Predicts latent diffusion σ_z(z)."""

    def __init__(self, latent_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.latent_dim = latent_dim
        self.n_tril = latent_dim * (latent_dim + 1) // 2
        neurons = [latent_dim] + hidden_dims + [self.n_tril]
        activations = [nn.Tanh()] * len(hidden_dims) + [None]
        self.net = FeedForwardNeuralNet(neurons, activations)
        # Pre-compute lower-triangular indices
        self.register_buffer(
            "_tril_rows",
            torch.tril_indices(latent_dim, latent_dim)[0],
        )
        self.register_buffer(
            "_tril_cols",
            torch.tril_indices(latent_dim, latent_dim)[1],
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, d)
        Returns:
            sigma: (B, d, d) lower-triangular diffusion matrix
        """
        B = z.shape[0]
        d = self.latent_dim
        raw = self.net(z)  # (B, n_tril)
        sigma = torch.zeros(B, d, d, device=z.device, dtype=z.dtype)
        sigma[:, self._tril_rows, self._tril_cols] = raw
        return sigma

    def covariance(self, z: Tensor) -> Tensor:
        """
        Compute Σ_z = σ_z σ_z^T (PSD by construction), symmetrized.

        Args:
            z: (B, d)
        Returns:
            Sigma_z: (B, d, d) symmetric PSD covariance
        """
        sigma = self.forward(z)
        cov = sigma @ sigma.mT
        return 0.5 * (cov + cov.mT)
