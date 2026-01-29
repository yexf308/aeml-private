import numpy as np
import pytest
import sympy as sp
import torch

from src.numeric.datagen import (
    convert_samples_to_torch,
    convert_to_torch,
    create_embedding_matrix,
    embed_data_with_qr_matrix,
    embed_dataset_with_qr_matrix,
    sample_from_manifold,
    train_test_split_tensors,
)
from src.numeric.datasets import DatasetBatch
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold


def _make_manifold() -> RiemannianManifold:
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, u * v])
    return RiemannianManifold(local_coord, chart)


def test_sample_from_manifold_shapes() -> None:
    man = _make_manifold()
    sde = ManifoldSDE(man)
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    batch = sample_from_manifold(sde, bounds, n_samples=8, seed=123)

    assert isinstance(batch, DatasetBatch)
    assert batch.samples.shape == (8, 3)
    assert batch.local_samples.shape == (8, 2)
    assert batch.mu.shape == (8, 3)
    assert batch.cov.shape == (8, 3, 3)
    assert batch.p.shape == (8, 3, 3)
    assert batch.weights.shape == (8,)
    assert batch.hessians.shape == (8, 3, 2, 2)

    assert torch.allclose(batch.weights.sum(), torch.tensor(1.0))


def test_convert_to_torch_dtype() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    t = convert_to_torch(arr)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.shape == (2, 2)


def test_convert_samples_to_torch_length_guard() -> None:
    samples = (np.zeros((1, 1)),) * 6
    with pytest.raises(ValueError):
        convert_samples_to_torch(samples)


def test_train_test_split_tensors_sizes() -> None:
    x = torch.zeros((10, 3))
    y = torch.ones((10, 2))
    train, test = train_test_split_tensors(x, y, test_size=0.3, seed=7)

    x_train, y_train = train
    x_test, y_test = test

    assert x_train.shape[0] == 7
    assert y_train.shape[0] == 7
    assert x_test.shape[0] == 3
    assert y_test.shape[0] == 3


def test_create_embedding_matrix_orthonormal_columns() -> None:
    embedding = create_embedding_matrix(embedding_dim=3, extrinsic_dim=5, embedding_seed=11)
    assert embedding.shape == (3, 5)
    gram = embedding @ embedding.T
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)


def test_embed_data_with_qr_matrix_shapes() -> None:
    n = 4
    extrinsic_dim = 5
    embedding_dim = 3
    intrinsic_dim = 2

    x = torch.randn(n, extrinsic_dim)
    mu = torch.randn(n, extrinsic_dim)
    cov = torch.randn(n, extrinsic_dim, extrinsic_dim)
    p = torch.randn(n, extrinsic_dim, extrinsic_dim)
    hessians = torch.randn(n, extrinsic_dim, intrinsic_dim, intrinsic_dim)
    embedding_matrix = create_embedding_matrix(embedding_dim, extrinsic_dim, embedding_seed=19)

    x_e, mu_e, cov_e, p_e, h_e = embed_data_with_qr_matrix(
        x, mu, cov, p, hessians, embedding_matrix
    )

    assert x_e.shape == (n, embedding_dim)
    assert mu_e.shape == (n, embedding_dim)
    assert cov_e.shape == (n, embedding_dim, embedding_dim)
    assert p_e.shape == (n, embedding_dim, embedding_dim)
    assert h_e.shape == (n, embedding_dim, intrinsic_dim, intrinsic_dim)


def test_embed_dataset_with_qr_matrix_shapes() -> None:
    n = 4
    extrinsic_dim = 5
    embedding_dim = 3
    intrinsic_dim = 2

    batch = DatasetBatch(
        samples=torch.randn(n, extrinsic_dim),
        local_samples=torch.randn(n, intrinsic_dim),
        mu=torch.randn(n, extrinsic_dim),
        cov=torch.randn(n, extrinsic_dim, extrinsic_dim),
        p=torch.randn(n, extrinsic_dim, extrinsic_dim),
        weights=torch.randn(n),
        hessians=torch.randn(n, extrinsic_dim, intrinsic_dim, intrinsic_dim),
    )
    embedding_matrix = create_embedding_matrix(embedding_dim, extrinsic_dim, embedding_seed=19)

    embedded = embed_dataset_with_qr_matrix(batch, embedding_matrix)

    assert embedded.samples.shape == (n, embedding_dim)
    assert embedded.mu.shape == (n, embedding_dim)
    assert embedded.cov.shape == (n, embedding_dim, embedding_dim)
    assert embedded.p.shape == (n, embedding_dim, embedding_dim)
    assert embedded.hessians.shape == (n, embedding_dim, intrinsic_dim, intrinsic_dim)
