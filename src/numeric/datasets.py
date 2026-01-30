from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DatasetBatch:
    samples: torch.Tensor
    local_samples: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    p: torch.Tensor
    weights: torch.Tensor
    hessians: torch.Tensor
    # Efficient tangent basis storage: U_d from SVD of P (optional)
    tangent_basis: torch.Tensor = None  # Shape: (batch, D, d) - top d eigenvectors

    def as_tuple(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.samples,
            self.local_samples,
            self.mu,
            self.cov,
            self.p,
            self.weights,
            self.hessians,
        )

    def compute_tangent_basis(self, intrinsic_dim: int) -> torch.Tensor:
        """
        Compute tangent basis U_d from projection matrix P via eigendecomposition.
        P = U_d @ U_d.T where U_d contains top d eigenvectors.

        Args:
            intrinsic_dim: The intrinsic dimension d

        Returns:
            tangent_basis: Shape (batch, D, d)
        """
        if self.tangent_basis is not None:
            return self.tangent_basis

        # P is symmetric, use eigendecomposition
        # eigenvalues are 0 or 1 for projection matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(self.p)
        # Take top d eigenvectors (largest eigenvalues, which should be ~1)
        self.tangent_basis = eigenvectors[..., -intrinsic_dim:]
        return self.tangent_basis

    @classmethod
    def from_tuple(cls, tensors: Tuple[torch.Tensor, ...]) -> "DatasetBatch":
        if len(tensors) != 7:
            raise ValueError("Expected a tuple of length 7.")
        return cls(*tensors)
