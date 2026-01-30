from torch import Tensor
import torch


def transform_covariance(cov: Tensor, jacobian: Tensor) -> Tensor:
    """
    Map a covariance matrix A to B A B^T where B is a Jacobian.
    """
    return torch.bmm(torch.bmm(jacobian, cov), jacobian.mT)


# The second order term coming from Ito's formula applied component-wise.
def ambient_quadratic_variation_drift(latent_covariance: Tensor, decoder_hessian: Tensor) -> Tensor:
    return torch.einsum("njk,nrkj->nr", latent_covariance, decoder_hessian)


def metric_tensor_from_jacobian(jacobian: Tensor) -> Tensor:
    return torch.bmm(jacobian.mT, jacobian)


def orthogonal_projection_from_jacobian(jacobian: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Compute orthogonal projection onto the tangent space.

    Args:
        jacobian: Jacobian tensor of shape (batch, output_dim, input_dim)
        eps: Small regularization term for numerical stability in matrix inversion

    Returns:
        Orthogonal projection tensor of shape (batch, output_dim, output_dim)
    """
    g = metric_tensor_from_jacobian(jacobian)
    # Add regularization for numerical stability
    g_reg = g + eps * torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
    g_inv = torch.linalg.inv(g_reg)
    return torch.bmm(jacobian, torch.bmm(g_inv, jacobian.mT))


def volume_measure_from_metric(metric: Tensor) -> Tensor:
    return torch.sqrt(torch.linalg.det(metric))
