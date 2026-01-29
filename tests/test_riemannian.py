import sympy as sp

from src.symbolic.riemannian import RiemannianManifold


def _make_manifold() -> RiemannianManifold:
    u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([u, v, u * v])
    return RiemannianManifold(local_coord, chart)


def test_implicit_function_shape() -> None:
    man = _make_manifold()
    implicit_fn = man.implicit_function()
    assert implicit_fn.shape == (1, 1)


def test_implicit_function_jacobian_shape() -> None:
    man = _make_manifold()
    jac = man.implicit_function_jacobian()
    assert jac is not None
    assert jac.shape == (1, 3)


def test_chart_jacobian_shape() -> None:
    man = _make_manifold()
    jac = man.chart_jacobian()
    assert jac.shape == (3, 2)


def test_metric_tensor_shape() -> None:
    man = _make_manifold()
    g = man.metric_tensor()
    assert g.shape == (2, 2)


def test_frames_shapes() -> None:
    man = _make_manifold()
    g_frame = man.g_orthonormal_frame()
    assert g_frame.shape == (2, 2)

    o_frame = man.orthonormal_frame()
    assert o_frame.shape == (3, 2)

    proj = man.orthogonal_projection()
    assert proj.shape == (3, 3)


def test_volume_density_is_scalar() -> None:
    man = _make_manifold()
    vol = man.volume_density()
    assert isinstance(vol, sp.Expr)


def test_christoffel_shape() -> None:
    man = _make_manifold()
    christoffel = man.christoffel_symbols()
    assert christoffel.shape == (2, 2, 2)


def test_compute_q_shape() -> None:
    man = _make_manifold()
    local_covariance = sp.eye(2)
    q = man.compute_q(local_covariance, ambient_dim=3)
    assert q.shape == (3, 1)
