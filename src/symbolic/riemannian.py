from typing import Callable, cast
from sympy import Matrix, MutableDenseNDimArray
import sympy as sp



def matrix_divergence(a: sp.Matrix, x: sp.Matrix):
    """
    Compute the matrix divergence, i.e. the Euclidean divergence applied row-wise to a matrix
    :param a:
    :param x:
    :return:
    """
    n, m = a.shape
    d = sp.zeros(n, 1)
    for i in range(n):
        for j in range(m):
            d[i] += sp.diff(a[i, j], x[j])
    return d

class RiemannianManifold:
    """
    A class representing a Riemannian manifold.

    This class provides methods to compute various geometric quantities and
    perform calculations on a Riemannian manifold defined by local coordinates
    and a chart.

    Attributes:
        local_coordinates (Matrix): The local coordinates of the manifold.
        chart (Matrix): The chart defining the manifold.
    """

    def __init__(self, local_coordinates: Matrix, chart: Matrix):
        """
        Initialize the RiemannianManifold.

        Args:
            local_coordinates (Matrix): The local coordinates of the manifold.
            chart (Matrix): The chart defining the manifold.
        """
        self.local_coordinates = local_coordinates
        self.chart = chart
        self.y = sp.symbols("y", real=True)

    def chart_jacobian(self) -> Matrix:
        """
        Compute the Jacobian of the chart.

        Returns:
            Matrix: The Jacobian matrix of the chart.
        """
        m = sp.simplify(self.chart.jacobian(self.local_coordinates))
        return Matrix(m)
    
    def chart_hessian(self):
        """
        Compute the Hessian of the chart.

        Returns:
            List[Matrix]: The list of Hessian matrices of the chart components.
        """
        hessians = []
        for i in range(len(self.chart)):
            hessian_i = sp.hessian(self.chart[i], self.local_coordinates)
            hessians.append(Matrix(sp.simplify(hessian_i)))
        return hessians

    def implicit_function(self):
        if len(self.chart) == 2:
            return sp.Matrix([self.y - self.chart[1]])
        elif len(self.chart) == 3:
            return sp.Matrix([self.y - self.chart[2]])
        else:
            raise ValueError("Unsupported chart dimension")

    def implicit_function_jacobian(self):
        if self.implicit_function() is not None:
            if len(self.chart) == 2:
                return self.implicit_function().jacobian([self.local_coordinates[0], self.y])
            elif len(self.chart) == 3:
                return self.implicit_function().jacobian([self.local_coordinates[0], self.local_coordinates[1], self.y])

    def metric_tensor(self) -> Matrix:
        """
        Compute the metric tensor of the manifold.

        Returns:
            Matrix: The metric tensor.
        """
        j = self.chart_jacobian()
        g = j.T * j
        return Matrix(sp.simplify(g))

    def volume_density(self) -> sp.Expr:
        """
        Compute the volume density of the manifold.

        Returns:
            sp.Expr: The volume density.
        """
        g = self.metric_tensor()
        return sp.simplify(sp.sqrt(sp.simplify(g.det())))

    def g_orthonormal_frame(self, method: str = "pow") -> Matrix:
        """
        Compute the g-orthonormal frame of the manifold.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The g-orthonormal frame.

        Raises:
            ValueError: If an invalid method is specified.
        """
        g = self.metric_tensor()
        g_inv = g.inv()
        if method == "pow":
            return Matrix(sp.simplify(g_inv.pow(1 / 2)))
        elif method == "svd":
            u, s, v = sp.Matrix(g_inv).singular_value_decomposition()
            sqrt_g_inv = u * sp.sqrt(s) * v
            return Matrix(sp.simplify(sqrt_g_inv))
        else:
            raise ValueError("argument 'method' must be 'pow' or 'svd'.")

    def orthonormal_frame(self, method: str = "pow") -> Matrix:
        """
        Compute the orthonormal frame of the manifold.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The orthonormal frame.
        """
        j = self.chart_jacobian()
        e = self.g_orthonormal_frame(method)
        return Matrix(sp.simplify(j * e))

    def orthogonal_projection(self, method: str = "pow") -> Matrix:
        """
        Compute the orthogonal projection of the manifold.

        Args:
            method (str): The method to use for computation. Either "pow" or "svd".

        Returns:
            Matrix: The orthogonal projection.
        """
        h = self.orthonormal_frame(method)
        return Matrix(sp.simplify(h * h.T))

    def manifold_divergence(self, f: Matrix) -> Matrix:
        """
        Compute the manifold divergence of a vector field.

        Args:
            f (Matrix): The vector field.

        Returns:
            sp.Expr: The manifold divergence.
        """
        vol_den = self.volume_density()
        scaled_field = Matrix(sp.simplify(vol_den * f))
        manifold_div = matrix_divergence(scaled_field, self.local_coordinates) / vol_den
        return Matrix(sp.simplify(manifold_div))

    def christoffel_symbols(self) -> MutableDenseNDimArray:
        """
        Compute the Christoffel symbols of the manifold.

        Returns:
            MutableDenseNDimArray: The Christoffel symbols.
        """
        g = self.metric_tensor()
        g_inv = sp.simplify(g.inv())
        n = len(self.local_coordinates)
        christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    term: sp.Expr = sp.S.Zero
                    for l in range(n):
                        diff_il_j = cast(sp.Expr, sp.diff(g[i, l], self.local_coordinates[j]))
                        diff_jl_i = cast(sp.Expr, sp.diff(g[j, l], self.local_coordinates[i]))
                        diff_ij_l = cast(sp.Expr, sp.diff(g[i, j], self.local_coordinates[l]))
                        g_inv_kl = cast(sp.Expr, g_inv[k, l])
                        increment = cast(sp.Expr, sp.Mul(
                            g_inv_kl,
                            sp.Add(diff_il_j, diff_jl_i, -diff_ij_l),
                            evaluate=True,
                        ))
                        term = cast(sp.Expr, sp.Add(term, increment, evaluate=True))
                    half_term = cast(sp.Expr, sp.Mul(term, sp.Rational(1, 2), evaluate=True))
                    christoffel[k, i, j] = sp.simplify(half_term)
        return christoffel
    
    def compute_q(self, local_covariance: Matrix, ambient_dim: int) -> Matrix:
        q = sp.zeros(ambient_dim, 1)
        for i in range(ambient_dim):
            hessian = sp.hessian(self.chart[i], self.local_coordinates)
            q[i] = sp.trace(local_covariance * hessian)
        return q


    def sympy_to_numpy(self, expr, local_coord=True) -> Callable:
        """
        Convert a sympy expression to a numpy-callable function.
        """
        # Separate cases for local-coordinate functions vs. functions on the ambient space.
        if local_coord:
            return sp.lambdify(self.local_coordinates, expr, modules='numpy')
        else: # this assumes one ambient coordinate y. We can generalize later if needed.
            return sp.lambdify(sp.Matrix([self.local_coordinates, self.y]), expr, modules='numpy')
