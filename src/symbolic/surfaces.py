import sympy as sp

def plane(u, v):
    return u+v

def paraboloid(u, v):
    return u**2+v**2

def hyperbolic_paraboloid(u,v):
    return v**2-u**2

def one_sheet_hyperboloid(u,v):
    return sp.sqrt(u**2+v**2-1)

def monkey_saddle(u,v):
    return u**3 - 3 * u * v ** 2

def gaussian_bump(u, v):
    return sp.exp(-u ** 2 -v ** 2)

def quartic_dome(u, v):
    """Quartic dome: f = (u^2+v^2) - (u^4+v^4)/2.

    Spatially varying Gaussian curvature: K=4 at the center (bowl),
    K=-8 at the edges (saddle).  Models an anharmonic potential well.
    """
    return (u**2 + v**2) - (u**4 + v**4) / 2

def sinusoidal(u, v):
    return sp.sin(u+v)


def half_torus_chart(u, v):
    """Half torus with major radius R=2, minor radius r=1.

    Parameterization:
        u in [-1,1] -> theta in [0, pi]  (upper half of tube cross-section)
        v in [-1,1] -> phi in [-pi, pi]  (around the hole)

    Gaussian curvature varies: K>0 on the outer edge, K=0 at the top,
    K<0 on the inner edge.  A single chart covers one half of the torus,
    making this a natural building block for a multi-chart atlas.
    """
    R, r = sp.Integer(2), sp.Integer(1)
    theta = sp.pi * (u + 1) / 2       # [0, pi]
    phi = sp.pi * v                    # [-pi, pi]
    x = (R + r * sp.cos(theta)) * sp.cos(phi)
    y = (R + r * sp.cos(theta)) * sp.sin(phi)
    z = r * sp.sin(theta)
    local_coord = sp.Matrix([u, v])
    chart = sp.Matrix([x, y, z])
    return local_coord, chart


def surface(fuv, u=None, v=None):
    """Build (local_coord, chart) for a Monge patch z = f(u,v)."""
    if u is None or v is None:
        u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    z = fuv(u,v)
    chart = sp.Matrix([u, v, z])
    return local_coord, chart
