from torch import Tensor
import torch
from torch.func import jvp, vmap, jacrev


def transform_covariance(cov: Tensor, jacobian: Tensor) -> Tensor:
    """
    Map a covariance matrix A to B A B^T where B is a Jacobian.
    """
    return torch.bmm(torch.bmm(jacobian, cov), jacobian.mT)


# The second order term coming from Ito's formula applied component-wise.
def ambient_quadratic_variation_drift(latent_covariance: Tensor, decoder_hessian: Tensor) -> Tensor:
    return torch.einsum("njk,nrkj->nr", latent_covariance, decoder_hessian)


def metric_tensor_from_jacobian(jacobian: Tensor) -> Tensor:
    return torch.bmm(jacobian.mT, jacobian)


def regularized_metric_inverse(g: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute (g + eps*I)^{-1} for numerical stability."""
    g_reg = g + eps * torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
    return torch.linalg.inv(g_reg)


def orthogonal_projection_from_jacobian(jacobian: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Compute orthogonal projection onto the tangent space.

    Args:
        jacobian: Jacobian tensor of shape (batch, output_dim, input_dim)
        eps: Small regularization term for numerical stability in matrix inversion

    Returns:
        Orthogonal projection tensor of shape (batch, output_dim, output_dim)
    """
    g = metric_tensor_from_jacobian(jacobian)
    g_inv = regularized_metric_inverse(g, eps)
    return torch.bmm(jacobian, torch.bmm(g_inv, jacobian.mT))


def volume_measure_from_metric(metric: Tensor) -> Tensor:
    return torch.sqrt(torch.linalg.det(metric))


# ============================================================================
# Hessian-free curvature computation via JVP
# ============================================================================

def hessian_vector_vector_product(func, z: Tensor, v: Tensor) -> Tensor:
    """
    Compute ∇²f(z)[v, v] (Hessian contracted with v twice) via double JVP.

    This avoids forming the full Hessian tensor.

    Args:
        func: Function to differentiate (decoder network)
        z: Input point, shape (d,) for single sample
        v: Direction vector, shape (d,)

    Returns:
        hvvp: ∇²f(z)[v,v], shape (D,) - second directional derivative
    """
    # First JVP: compute f(z) and ∂f/∂z · v
    def f_single(z_in):
        return func(z_in.unsqueeze(0)).squeeze(0)

    # ∂f/∂z · v via forward-mode AD
    _, jvp1 = jvp(f_single, (z,), (v,))

    # Second JVP: differentiate (∂f/∂z · v) w.r.t. z in direction v
    def jvp1_func(z_in):
        _, out = jvp(f_single, (z_in,), (v,))
        return out

    _, hvvp = jvp(jvp1_func, (z,), (v,))

    return hvvp


def curvature_drift_hessian_free(
    decoder_func,
    z: Tensor,
    local_cov: Tensor,
    normal_proj: Tensor,
) -> Tensor:
    """
    Compute the Λ-weighted mean-curvature vector using Hessian-free JVP.

    Computes: (1/2) * Σᵢⱼ Λᵢⱼ (I-P) ∂²φ/∂zᵢ∂zⱼ

    Using eigendecomposition of local covariance Λ = Σₖ λₖ eₖeₖᵀ:
        = (1/2) * Σₖ λₖ (I-P) ∇²φ(eₖ, eₖ)

    Each ∇²φ(eₖ, eₖ) is computed via double JVP, avoiding the O(Dd²) Hessian tensor.

    Args:
        decoder_func: Decoder network callable
        z: Latent points, shape (B, d)
        local_cov: Local covariance matrices, shape (B, d, d)
        normal_proj: Normal projection (I-P), shape (B, D, D)

    Returns:
        curvature_drift: (1/2) * II : Λ, shape (B, D)

    Complexity: O(d × JVP_cost) time, O(Dd) memory (no Hessian tensor)
    """
    B, d = z.shape
    D = normal_proj.shape[-1]
    device = z.device
    dtype = z.dtype

    # Eigendecompose local covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(local_cov)  # (B, d), (B, d, d)

    # Vectorized approach: compute all B×d Hessian-vector-vector products
    result = torch.zeros(B, D, device=device, dtype=dtype)

    # For each eigenvector direction k
    for k in range(d):
        e_k = eigenvectors[:, :, k]  # (B, d) - k-th eigenvector for all samples
        lam_k = eigenvalues[:, k]    # (B,) - k-th eigenvalue for all samples

        # Compute ∇²φ(e_k, e_k) via double JVP for all samples
        hvvp = hessian_vector_vector_product_batch(decoder_func, z, e_k)  # (B, D)

        # Project to normal space and weight by eigenvalue
        normal_hvvp = torch.bmm(normal_proj, hvvp.unsqueeze(-1)).squeeze(-1)  # (B, D)
        result = result + lam_k.unsqueeze(-1) * normal_hvvp

    return 0.5 * result


def hessian_vector_vector_product_batch(func, z: Tensor, v: Tensor) -> Tensor:
    """
    Batched computation of ∇²f(z)[v, v] via double JVP.

    Args:
        func: Function to differentiate (decoder network)
        z: Input points, shape (B, d)
        v: Direction vectors, shape (B, d)

    Returns:
        hvvp: ∇²f(z)[v,v] for each sample, shape (B, D)
    """
    B, d = z.shape

    def hvvp_single(z_single, v_single):
        """Compute HVVP for single sample."""
        def f_single(z_in):
            return func(z_in.unsqueeze(0)).squeeze(0)

        def jvp_func(z_in):
            _, out = jvp(f_single, (z_in,), (v_single,))
            return out

        _, hvvp = jvp(jvp_func, (z_single,), (v_single,))
        return hvvp

    # Vectorize over batch
    return vmap(hvvp_single)(z, v)


def curvature_drift_hessian_free_ambient(
    decoder_func,
    z: Tensor,
    ambient_cov: Tensor,
    encoder_jacobian: Tensor,
    normal_proj: Tensor,
) -> Tensor:
    """
    Compute curvature drift from AMBIENT covariance using Hessian-free JVP.

    This version directly uses ambient covariance eigenvectors, mapping them
    to latent space via the encoder Jacobian.

    Computes: (1/2) * II : Λ = (1/2) * Σₖ λₖ (I-P) ∇²φ(vₖ, vₖ)

    where:
    - Λ = Σₖ λₖ uₖuₖᵀ is the eigendecomposition of ambient covariance
    - vₖ = ∇ψ · uₖ maps ambient eigenvector to latent direction
    - ∇²φ(vₖ, vₖ) is computed via double JVP (no explicit Hessian)

    Args:
        decoder_func: Decoder network callable
        z: Latent points, shape (B, d)
        ambient_cov: Ambient covariance matrices, shape (B, D, D)
        encoder_jacobian: Encoder Jacobian ∇ψ, shape (B, d, D)
        normal_proj: Normal projection (I-P), shape (B, D, D)

    Returns:
        curvature_drift: (1/2) * II : Λ, shape (B, D)
    """
    B, d = z.shape
    D = ambient_cov.shape[-1]
    device = z.device
    dtype = z.dtype

    # Eigendecompose ambient covariance (rank-d, so only d non-zero eigenvalues)
    eigenvalues, eigenvectors = torch.linalg.eigh(ambient_cov)  # (B, D), (B, D, D)

    # Take top-d eigenvalues and eigenvectors
    lambda_k = eigenvalues[..., -d:]  # (B, d)
    u_k = eigenvectors[..., -d:]  # (B, D, d)

    # Map ambient eigenvectors to latent directions: v_k = ∇ψ @ u_k
    # encoder_jacobian: (B, d, D), u_k: (B, D, d)
    v_k = torch.bmm(encoder_jacobian, u_k)  # (B, d, d)

    # Compute Σₖ λₖ (I-P) ∇²φ(vₖ, vₖ) via JVP
    result = torch.zeros(B, D, device=device, dtype=dtype)

    def compute_single_sample(z_single, v_k_single, lambda_k_single, normal_proj_single):
        """Compute curvature drift for a single sample."""
        curvature_sum = torch.zeros(D, device=device, dtype=dtype)

        for k in range(d):
            v = v_k_single[:, k]  # (d,)
            lam = lambda_k_single[k]  # scalar

            # Compute ∇²φ(v, v) via double JVP
            hvvp = hessian_vector_vector_product(decoder_func, z_single, v)  # (D,)

            # Project to normal space and weight by eigenvalue
            normal_hvvp = normal_proj_single @ hvvp  # (D,)
            curvature_sum = curvature_sum + lam * normal_hvvp

        return 0.5 * curvature_sum

    # Process each sample
    for b in range(B):
        result[b] = compute_single_sample(
            z[b], v_k[b], lambda_k[b], normal_proj[b]
        )

    return result


def curvature_drift_explicit(
    decoder_hessian: Tensor,
    local_cov: Tensor,
    normal_proj: Tensor,
) -> Tensor:
    """
    Compute the Λ-weighted mean-curvature vector using explicit Hessian.

    This is the original O(Dd²) memory approach.

    Args:
        decoder_hessian: Full Hessian tensor, shape (B, D, d, d)
        local_cov: Local covariance (transformed), shape (B, d, d)
        normal_proj: Normal projection (I-P), shape (B, D, D)

    Returns:
        curvature_drift: (1/2) * II : Λ, shape (B, D)
    """
    # Contract Hessian with local covariance: Σᵢⱼ Λᵢⱼ ∂²φ/∂zᵢ∂zⱼ
    q = ambient_quadratic_variation_drift(local_cov, decoder_hessian)  # (B, D)

    # Project to normal space
    normal_q = torch.bmm(normal_proj, q.unsqueeze(-1)).squeeze(-1)  # (B, D)

    return 0.5 * normal_q


def curvature_drift_explicit_full(
    decoder_hessian: Tensor,
    local_cov: Tensor,
) -> Tensor:
    """
    Compute the full-space (unprojected) Ito correction using explicit Hessian.

    Unlike curvature_drift_explicit, this does NOT project onto the normal space.
    The result is the full ambient-space Ito correction vector:
        (1/2) * Σᵢⱼ Λᵢⱼ ∂²φ/∂zᵢ∂zⱼ

    Note: this quantity is chart-dependent. Only the sum Dφ·b_z + q is
    chart-invariant. See design doc for discussion.

    Args:
        decoder_hessian: Full Hessian tensor, shape (B, D, d, d)
        local_cov: Local covariance (transformed), shape (B, d, d)

    Returns:
        ito_correction: (1/2) * tr(Λ·H), shape (B, D)
    """
    q = ambient_quadratic_variation_drift(local_cov, decoder_hessian)  # (B, D)
    return 0.5 * q


def curvature_drift_hessian_free_full(
    decoder_func,
    z: Tensor,
    local_cov: Tensor,
) -> Tensor:
    """
    Compute the full-space (unprojected) Ito correction using Hessian-free JVP.

    Like curvature_drift_hessian_free but without the normal projection step.
    Uses Proposition 8: Σᵢⱼ Λᵢⱼ H_{r,ij} = Σₖ λₖ ∇²φ_r(eₖ, eₖ)

    Note: this quantity is chart-dependent. See curvature_drift_explicit_full.

    Args:
        decoder_func: Decoder network callable
        z: Latent points, shape (B, d)
        local_cov: Local covariance matrices, shape (B, d, d)

    Returns:
        ito_correction: (1/2) * tr(Λ·H), shape (B, D)
    """
    B, d = z.shape

    # Eigendecompose local covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(local_cov)  # (B, d), (B, d, d)

    # Determine D from a probe forward pass
    with torch.no_grad():
        D = decoder_func(z[:1]).shape[-1]

    result = torch.zeros(B, D, device=z.device, dtype=z.dtype)

    for k in range(d):
        e_k = eigenvectors[:, :, k]  # (B, d)
        lam_k = eigenvalues[:, k]    # (B,)

        hvvp = hessian_vector_vector_product_batch(decoder_func, z, e_k)  # (B, D)
        # NO normal projection — full ambient vector
        result = result + lam_k.unsqueeze(-1) * hvvp

    return 0.5 * result
