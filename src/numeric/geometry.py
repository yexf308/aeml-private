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


def orthogonal_projection_from_jacobian(jacobian: Tensor) -> Tensor:
    g = metric_tensor_from_jacobian(jacobian)
    g_inv = torch.linalg.inv(g)
    return torch.bmm(jacobian, torch.bmm(g_inv, jacobian.mT))


def volume_measure_from_metric(metric: Tensor) -> Tensor:
    return torch.sqrt(torch.linalg.det(metric))
