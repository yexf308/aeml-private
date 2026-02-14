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


def test_ambient_diffusion_equals_dphi_times_local_diffusion() -> None:
    """Verify Λ^{1/2} = Dφ · σ (Itô's formula)."""
    man = _make_manifold()
    u, v = man.local_coordinates
    local_drift = sp.Matrix([u, -v])
    local_diffusion = sp.Matrix([[1, 0], [0, 1]])
    sde = ManifoldSDE(man, local_drift=local_drift, local_diffusion=local_diffusion)

    dphi = man.chart_jacobian()
    expected = sp.simplify(dphi * local_diffusion)
    diff = sp.simplify(sde.ambient_diffusion - expected)
    assert diff == sp.zeros(3, 2)


def test_ambient_covariance_equals_dphi_sigma_dphi_T() -> None:
    """Verify Λ = Dφ Σ Dφᵀ."""
    man = _make_manifold()
    u, v = man.local_coordinates
    local_drift = sp.Matrix([0, 0])
    local_diffusion = sp.eye(2)
    sde = ManifoldSDE(man, local_drift=local_drift, local_diffusion=local_diffusion)

    dphi = man.chart_jacobian()
    sigma = local_diffusion * local_diffusion.T
    expected = sp.simplify(dphi * sigma * dphi.T)
    diff = sp.simplify(sde.ambient_covariance - expected)
    assert diff == sp.zeros(3, 3)


def test_compute_q_on_plane() -> None:
    """On a plane φ(u,v) = (u, v, au+bv), all Hessians are zero so q=0."""
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, 3*u + 2*v])  # plane
    man = RiemannianManifold(local_coord, chart)
    sde = ManifoldSDE(man, local_drift=sp.Matrix([0, 0]), local_diffusion=sp.eye(2))

    q = sde.compute_q()
    assert sp.simplify(q) == sp.zeros(3, 1)


def test_compute_q_on_paraboloid() -> None:
    """On the paraboloid φ(u,v) = (u, v, u²+v²), q should be nonzero."""
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, u**2 + v**2])
    man = RiemannianManifold(local_coord, chart)

    # With identity covariance, q^i = Tr(I · H_i)
    # H_1 = 0 (φ¹=u), H_2 = 0 (φ²=v), H_3 = diag(2,2) (φ³=u²+v²)
    # So q = (0, 0, Tr(diag(2,2))) = (0, 0, 4)
    sde = ManifoldSDE(man, local_drift=sp.Matrix([0, 0]), local_diffusion=sp.eye(2))
    q = sde.compute_q()

    assert sp.simplify(q[0]) == 0
    assert sp.simplify(q[1]) == 0
    assert sp.simplify(q[2]) == 4


def test_ambient_drift_ito_formula() -> None:
    """Verify b = Dφ μ + ½q (Itô's formula for the drift)."""
    man = _make_manifold()
    u, v = man.local_coordinates
    local_drift = sp.Matrix([u, -v])
    local_diffusion = sp.eye(2)
    sde = ManifoldSDE(man, local_drift=local_drift, local_diffusion=local_diffusion)

    dphi = man.chart_jacobian()
    q = sde.compute_q()
    expected_drift = sp.simplify(dphi * local_drift + q / 2)
    diff = sp.simplify(sde.ambient_drift - expected_drift)
    assert diff == sp.zeros(3, 1)


def test_ambient_covariance_range_equals_tangent_space() -> None:
    """The rank of Λ should equal the intrinsic dimension (tangent space)."""
    man = _make_manifold()
    sde = ManifoldSDE(man)

    # Evaluate at a generic point to check rank
    u, v = man.local_coordinates
    cov_at_point = sde.ambient_covariance.subs([(u, 1), (v, 1)])
    assert cov_at_point.rank() == 2  # intrinsic dim


def test_bm_drift_is_mean_curvature_vector() -> None:
    """For Brownian motion on (M,g), the ambient drift should be ½ times
    the g-mean-curvature vector (which is normal to M)."""
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, u**2 + v**2])  # paraboloid
    man = RiemannianManifold(local_coord, chart)

    # Default BM: drift = div(g^{-1}/2), diffusion = g^{-1/2}
    sde = ManifoldSDE(man)

    # The tangent projection P and normal projection N = I - P
    dphi = man.chart_jacobian()
    g = man.metric_tensor()
    ginv = g.inv()
    P = sp.simplify(dphi * ginv * dphi.T)

    # For BM, the tangent part of the drift should vanish (P · b = Dφ · μ_tangent)
    # and the normal part is ½ c_g^α n_α (mean curvature vector)
    # Specifically, for BM: b = ½ q where q^i = Tr(g^{-1} H_i)
    # so the tangent projection of b = Dφ · μ where μ is the BM drift
    # and the normal projection is ½ N q
    N = sp.eye(3) - P
    normal_drift = sp.simplify(N * sde.ambient_drift)

    # At the origin, the paraboloid is locally flat, so normal drift should be simple
    normal_at_origin = normal_drift.subs([(u, 0), (v, 0)])
    # At origin g = I_2, so q = (0, 0, Tr(I · diag(2,2))) = (0, 0, 4)
    # N at origin = diag(0, 0, 1) (since tangent plane is x-y plane)
    # So normal drift = ½ * (0, 0, 4) = (0, 0, 2)
    assert sp.simplify(normal_at_origin[0]) == 0
    assert sp.simplify(normal_at_origin[1]) == 0
    assert sp.simplify(normal_at_origin[2]) == 2
