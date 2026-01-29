from ..symbolic.manifold_sdes import ManifoldSDE
from .sampler import ImportanceSampler
from .datasets import DatasetBatch
from typing import Tuple
import numpy as np
import sympy as sp
import torch

def sample_from_manifold(manifold_sde: ManifoldSDE, bounds, n_samples=1000, seed=None) -> DatasetBatch:
    """
    Sample points from a Riemannian manifold using importance sampling.

    Args:
        manifold_sde: An instance of ManifoldSDE defining the manifold/dynamics
        bounds: A list of tuples defining the sampling bounds for each local coordinate.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        what: what numerics package to return samples in, either "torch" or "numpy".
    Returns:
        ambient_samples: Samples mapped to the ambient space.
        local_samples: Samples in the local coordinates.
        weights: Importance weights for the samples.
    """
    # Compute the volume density function.
    vol_density = manifold_sde.manifold.volume_density()
    
    np_vol = manifold_sde.manifold.sympy_to_numpy(vol_density)
    np_phi = manifold_sde.manifold.sympy_to_numpy(manifold_sde.manifold.chart)
    np_ambient_drift = manifold_sde.manifold.sympy_to_numpy(manifold_sde.ambient_drift)
    # np_ambient_diffusion = manifold_sde.manifold.sympy_to_numpy(manifold_sde.ambient_diffusion)
    np_ambient_covariance = manifold_sde.manifold.sympy_to_numpy(manifold_sde.ambient_covariance)
    np_orthogonal_projection = manifold_sde.manifold.sympy_to_numpy(manifold_sde.manifold.orthogonal_projection())
    np_phi_hessian = [manifold_sde.manifold.sympy_to_numpy(sp.hessian(manifold_sde.manifold.chart[i], manifold_sde.manifold.local_coordinates)) for i in range(manifold_sde.extrinsic_dim)]
    # Flatten bounds for ImportanceSampler
    flat_bounds = [b for bound in bounds for b in bound]
    
    # Sample from the manifold in local-coordinates.
    sampler = ImportanceSampler(np_vol, *flat_bounds)
    local_samples, weights = sampler.sample(n_samples, seed=seed)
    
    # Map to ambient space.
    ambient_samples = np.squeeze(
        np.array([np_phi(*sample) for sample in local_samples]),
        axis=-1,
    )
    extrinsic_drifts = np.squeeze(
        np.array([np_ambient_drift(*sample) for sample in local_samples]),
        axis=-1,
    )
    extrinsic_covariances = np.array([np_ambient_covariance(*sample) for sample in local_samples])
    orthogonal_projections = np.array([np_orthogonal_projection(*sample) for sample in local_samples])
    hessians = np.array([[np_phi_hessian[i](*sample) for i in range(manifold_sde.extrinsic_dim)] for sample in local_samples])
    return convert_samples_to_torch(
        (ambient_samples, local_samples, extrinsic_drifts, extrinsic_covariances, orthogonal_projections, weights, hessians)
    )


def convert_to_torch(array: np.ndarray, device='cpu') -> torch.Tensor:
    """Convert a numpy array to a PyTorch tensor on the specified device."""
    return torch.tensor(array, dtype=torch.float32, device=device)

def convert_samples_to_torch(samples_tuple, device='cpu') -> DatasetBatch:
    """Convert all elements in a tuple of numpy arrays to PyTorch tensors."""
    torch_tuple = tuple([convert_to_torch(arr, device) for arr in samples_tuple])
    return DatasetBatch.from_tuple(torch_tuple)

def train_test_split_tensors(*tensors, test_size=0.2, seed=17) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    n_total = tensors[0].shape[0]
    n_train = int(n_total * (1 - test_size))
    
    # Generate shuffled indices
    torch.manual_seed(seed)
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split all tensors
    train_tensors = tuple(tensor[train_indices] for tensor in tensors)
    test_tensors = tuple(tensor[test_indices] for tensor in tensors)
    
    return train_tensors, test_tensors


def train_test_split_dataset(dataset: DatasetBatch, test_size=0.2, seed=17) -> Tuple[DatasetBatch, DatasetBatch]:
    train_tensors, test_tensors = train_test_split_tensors(*dataset.as_tuple(), test_size=test_size, seed=seed)
    return DatasetBatch.from_tuple(train_tensors), DatasetBatch.from_tuple(test_tensors)


def embed_dataset_with_qr_matrix(dataset: DatasetBatch, embedding_matrix) -> DatasetBatch:
    x_e, mu_e, cov_e, p_e, h_e = embed_data_with_qr_matrix(
        dataset.samples,
        dataset.mu,
        dataset.cov,
        dataset.p,
        dataset.hessians,
        embedding_matrix,
    )
    return DatasetBatch(
        samples=x_e,
        local_samples=dataset.local_samples,
        mu=mu_e,
        cov=cov_e,
        p=p_e,
        weights=dataset.weights,
        hessians=h_e,
    )

def embed_data_with_qr_matrix(x, mu, cov, p, hessians, embedding_matrix) -> Tuple[torch.Tensor,...]:
    # Embed every object
    x_embed = x @ embedding_matrix.T
    mu_embed = mu @ embedding_matrix.T
    cov_embed = embedding_matrix @ cov @ embedding_matrix.T
    p_embed = embedding_matrix @ p @ embedding_matrix.T
    hessian_embed = torch.einsum('ai, tibc -> tabc', embedding_matrix, hessians)
    return x_embed, mu_embed, cov_embed, p_embed, hessian_embed

def create_embedding_matrix(embedding_dim, extrinsic_dim, embedding_seed=17) -> torch.Tensor:
    # Embedding matrix
    rng = np.random.default_rng(seed=embedding_seed)
    base = rng.standard_normal(size=(extrinsic_dim, embedding_dim))
    base = torch.tensor(base, dtype=torch.float32)
    q, _ = torch.linalg.qr(base)
    return q.T
