from typing import Optional

import sympy as sp
from sympy import Matrix

from .riemannian import RiemannianManifold

class ManifoldSDE(object):
    def __init__(self, manifold: RiemannianManifold, local_drift: Optional[Matrix] = None, local_diffusion: Optional[Matrix] = None):
        self.manifold = manifold
        self.local_drift = local_drift
        self.local_diffusion = local_diffusion
        self.create_local_bm_coefficients() # Ensures local_drift and local_diffusion are set.
        if self.local_diffusion is not None:
            self.local_covariance = self.local_diffusion * self.local_diffusion.T
        else:
            self.local_covariance = None
        self.intrinsic_dim = len(self.manifold.local_coordinates)
        self.extrinsic_dim = len(self.manifold.chart)
        self.create_ambient_coefficients() # Computes the ambient drift and diffusion.
        self.ambient_covariance = sp.simplify(self.ambient_diffusion * self.ambient_diffusion.T)

    def create_local_bm_coefficients(self):
        if self.local_drift is None:
            g = self.manifold.metric_tensor()
            g_inv = g.inv()
            self.local_drift = self.manifold.manifold_divergence(g_inv/2)
            if self.local_diffusion is None:
                self.local_diffusion = Matrix(sp.simplify(g_inv.pow(1/2)))
        elif self.local_diffusion is None:
            self.local_diffusion = self.manifold.g_orthonormal_frame()
        return None
            
    def compute_q(self):
        q = sp.zeros(self.extrinsic_dim, 1)
        for i in range(self.extrinsic_dim):
            hessian = sp.hessian(self.manifold.chart[i], self.manifold.local_coordinates)
            q[i] = sp.trace(self.local_covariance * hessian)
        return q
    
    def create_ambient_coefficients(self):
        dphi = self.manifold.chart_jacobian()
        q = self.compute_q()
        self.ambient_drift = sp.simplify(dphi * self.local_drift + q/2)
        self.ambient_diffusion = sp.simplify(dphi * self.local_diffusion)
        return None
