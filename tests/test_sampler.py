import numpy as np
import pytest

from src.numeric.sampler import ImportanceSampler


def test_sampler_1d_shapes_and_weights() -> None:
    def h1d(x):
        return np.exp(-(x - 2) ** 2)

    sampler = ImportanceSampler(h1d, 0, 4)
    samples, weights = sampler.sample(500, seed=123)

    assert samples.shape == (500, 1)
    assert weights.shape == (500,)
    assert np.allclose(weights.sum(), 1.0)
    assert np.all((samples >= 0) & (samples <= 4))


def test_sampler_2d_shapes_and_weights() -> None:
    def h2d(x, y):
        return np.exp(-(x - 2) ** 2 - (y - 2) ** 2)

    sampler = ImportanceSampler(h2d, 0, 4, 0, 4)
    samples, weights = sampler.sample(500, seed=123)

    assert samples.shape == (500, 2)
    assert weights.shape == (500,)
    assert np.allclose(weights.sum(), 1.0)
    assert np.all((samples >= 0) & (samples <= 4))


def test_q_returns_vector_for_vector_input() -> None:
    def h1d(x):
        return np.exp(-(x - 2) ** 2)

    sampler = ImportanceSampler(h1d, 0, 4)
    x = np.array([0.0, 1.0, 2.0])
    q = sampler.q(x)
    assert q.shape == x.shape
    assert np.allclose(q, q[0])


def test_invalid_dimension_raises() -> None:
    def h3d(x, y, z):
        return np.exp(-(x - 2) ** 2 - (y - 2) ** 2 - (z - 2) ** 2)

    with pytest.raises(ValueError):
        ImportanceSampler(h3d, 0, 4, 0, 4, 0, 4)
