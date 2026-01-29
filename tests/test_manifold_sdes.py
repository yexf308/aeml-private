import sympy as sp

from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold


def _make_manifold() -> RiemannianManifold:
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, u * v])
    return RiemannianManifold(local_coord, chart)


def test_sde_shapes_from_defaults() -> None:
    man = _make_manifold()
    sde = ManifoldSDE(man)

    assert sde.local_drift is not None
    assert sde.local_diffusion is not None
    assert sde.local_covariance is not None

    assert sde.local_drift.shape == (2, 1)
    assert sde.local_diffusion.shape == (2, 2)
    assert sde.local_covariance.shape == (2, 2)

    assert sde.ambient_drift.shape == (3, 1)
    assert sde.ambient_diffusion.shape == (3, 2)
    assert sde.ambient_covariance.shape == (3, 3)

    assert sde.intrinsic_dim == 2
    assert sde.extrinsic_dim == 3


def test_sde_uses_provided_coefficients() -> None:
    man = _make_manifold()
    u, v = man.local_coordinates
    local_drift = sp.Matrix([u, v])
    local_diffusion = sp.eye(2)

    sde = ManifoldSDE(man, local_drift=local_drift, local_diffusion=local_diffusion)

    assert sde.local_drift == local_drift
    assert sde.local_diffusion == local_diffusion
    assert sde.local_covariance is not None
    assert sde.local_covariance.shape == (2, 2)
    assert sde.ambient_drift.shape == (3, 1)
    assert sde.ambient_diffusion.shape == (3, 2)


def test_sde_fills_missing_diffusion() -> None:
    man = _make_manifold()
    u, v = man.local_coordinates
    local_drift = sp.Matrix([u, v])

    sde = ManifoldSDE(man, local_drift=local_drift, local_diffusion=None)

    assert sde.local_drift == local_drift
    assert sde.local_diffusion is not None
    assert sde.local_diffusion.shape == (2, 2)
