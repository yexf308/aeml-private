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

def sinusoidal(u, v):
    return sp.sin(u+v)

def surface(fuv, u=None, v=None):
    if u is None or v is None:
        u, v = sp.symbols("u v", real=True)
    local_coord = sp.Matrix([u, v])
    z = fuv(u,v)
    chart = sp.Matrix([u, v, z])
    return local_coord, chart
