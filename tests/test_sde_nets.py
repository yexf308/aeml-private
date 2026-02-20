"""
Tests for the data-driven latent SDE pipeline components.

Covers:
- DriftNet / DiffusionNet forward shapes and properties
- tangential_drift_loss / ambient_diffusion_loss correctness
- Gradient isolation: each loss only trains its target net
- Numerical conventions: halved q, stable pseudoinverse, symmetry, P_hat idempotency
"""
import torch
import torch.nn as nn
import pytest

from src.numeric.autoencoders import AutoEncoder
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_losses import tangential_drift_loss, ambient_diffusion_loss
from src.numeric.geometry import (
    curvature_drift_explicit_full,
    ambient_quadratic_variation_drift,
    regularized_metric_inverse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ae():
    """Small autoencoder for testing."""
    return AutoEncoder(
        extrinsic_dim=3, intrinsic_dim=2, hidden_dims=[8],
        encoder_act=nn.Tanh(), decoder_act=nn.Tanh(), tie_weights=False,
    )


@pytest.fixture
def drift_net():
    return DriftNet(latent_dim=2, hidden_dims=[8, 8])


@pytest.fixture
def diffusion_net():
    return DiffusionNet(latent_dim=2, hidden_dims=[8, 8])


@pytest.fixture
def sample_data():
    """Generate sample data: (x, v, Lambda) in ambient space."""
    B, D = 6, 3
    x = torch.randn(B, D)
    v = torch.randn(B, D)
    L = torch.randn(B, D, D) * 0.1
    Lambda = L @ L.mT + 0.01 * torch.eye(D).unsqueeze(0)
    return x, v, Lambda


# ---------------------------------------------------------------------------
# DriftNet tests
# ---------------------------------------------------------------------------

class TestDriftNet:
    def test_forward_shape(self, drift_net):
        z = torch.randn(4, 2)
        out = drift_net(z)
        assert out.shape == (4, 2)

    def test_output_is_finite(self, drift_net):
        z = torch.randn(8, 2)
        out = drift_net(z)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# DiffusionNet tests
# ---------------------------------------------------------------------------

class TestDiffusionNet:
    def test_forward_shape(self, diffusion_net):
        z = torch.randn(4, 2)
        sigma = diffusion_net(z)
        assert sigma.shape == (4, 2, 2)

    def test_forward_is_lower_triangular(self, diffusion_net):
        z = torch.randn(4, 2)
        sigma = diffusion_net(z)
        # Upper-triangular part (excluding diagonal) should be zero
        for i in range(2):
            for j in range(i + 1, 2):
                assert (sigma[:, i, j] == 0).all()

    def test_covariance_shape(self, diffusion_net):
        z = torch.randn(4, 2)
        cov = diffusion_net.covariance(z)
        assert cov.shape == (4, 2, 2)

    def test_covariance_is_psd(self, diffusion_net):
        z = torch.randn(8, 2)
        cov = diffusion_net.covariance(z)
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-6).all(), f"Min eigenvalue: {eigenvalues.min().item()}"

    def test_covariance_is_symmetric(self, diffusion_net):
        z = torch.randn(8, 2)
        cov = diffusion_net.covariance(z)
        assert torch.allclose(cov, cov.mT, atol=1e-6)


# ---------------------------------------------------------------------------
# Tangential drift loss tests
# ---------------------------------------------------------------------------

class TestTangentialDriftLoss:
    def test_loss_runs(self, ae, drift_net, sample_data):
        """Loss computes without error."""
        x, v, Lambda = sample_data
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = ae.encoder(x).detach()
        loss = tangential_drift_loss(ae.decoder, drift_net, z, v, Lambda)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flows_to_drift_net_only(self, ae, drift_net, diffusion_net, sample_data):
        """Gradient should flow to drift_net only, not AE or diffusion_net."""
        x, v, Lambda = sample_data
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = ae.encoder(x).detach()
        loss = tangential_drift_loss(ae.decoder, drift_net, z, v, Lambda)
        loss.backward()

        # drift_net should have gradients
        drift_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in drift_net.parameters()
        )
        assert drift_has_grad, "drift_net should receive gradients"

        # AE should NOT have gradients
        for p in ae.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, \
                "AE should not receive gradients during drift training"

        # diffusion_net should NOT have gradients (wasn't used)
        for p in diffusion_net.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_normal_component_ignored(self, ae, drift_net, sample_data):
        """Adding a normal component to v shouldn't change the loss significantly."""
        x, v, Lambda = sample_data
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = ae.encoder(x).detach()
        dphi = ae.decoder.jacobian_network(z)
        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        P_hat = dphi @ ginv @ dphi.mT
        P_hat = 0.5 * (P_hat + P_hat.mT)

        # Construct normal perturbation
        D = v.shape[1]
        I_mat = torch.eye(D).unsqueeze(0)
        N_hat = I_mat - P_hat
        normal_noise = (N_hat @ torch.randn_like(v).unsqueeze(-1)).squeeze(-1) * 10.0

        loss_clean = tangential_drift_loss(ae.decoder, drift_net, z, v, Lambda)
        loss_noisy = tangential_drift_loss(ae.decoder, drift_net, z, v + normal_noise, Lambda)

        # Losses should be close since the loss projects onto tangent space
        assert torch.allclose(loss_clean, loss_noisy, rtol=0.05, atol=1e-4), \
            f"Normal perturbation changed loss: {loss_clean.item():.6f} vs {loss_noisy.item():.6f}"


# ---------------------------------------------------------------------------
# Ambient diffusion loss tests
# ---------------------------------------------------------------------------

class TestAmbientDiffusionLoss:
    def test_loss_runs(self, ae, diffusion_net, sample_data):
        """Loss computes without error."""
        x, _, Lambda = sample_data
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = ae.encoder(x).detach()
        loss = ambient_diffusion_loss(diffusion_net, z, Lambda, decoder=ae.decoder)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_with_precomputed_dphi(self, ae, diffusion_net, sample_data):
        """Loss works with precomputed Jacobian."""
        x, _, Lambda = sample_data
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = ae.encoder(x).detach()
        dphi = ae.decoder.jacobian_network(z).detach()
        loss = ambient_diffusion_loss(diffusion_net, z, Lambda, dphi=dphi)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flows_to_diffusion_net_only(self, ae, drift_net, diffusion_net, sample_data):
        """Gradient should flow to diffusion_net only."""
        x, _, Lambda = sample_data
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = ae.encoder(x).detach()
        loss = ambient_diffusion_loss(diffusion_net, z, Lambda, decoder=ae.decoder)
        loss.backward()

        # diffusion_net should have gradients
        diff_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in diffusion_net.parameters()
        )
        assert diff_has_grad, "diffusion_net should receive gradients"

        # AE should NOT have gradients
        for p in ae.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

        # drift_net should NOT have gradients (wasn't used)
        for p in drift_net.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# Numerical convention tests
# ---------------------------------------------------------------------------

class TestNumericalConventions:
    def test_halved_q_convention(self):
        """curvature_drift_explicit_full should return halved Ito correction."""
        B, D, d = 4, 5, 2
        d2phi = torch.randn(B, D, d, d)
        L = torch.randn(B, d, d)
        local_cov = L @ L.mT + 0.1 * torch.eye(d).unsqueeze(0)

        q = curvature_drift_explicit_full(d2phi, local_cov)
        raw = ambient_quadratic_variation_drift(local_cov, d2phi)

        assert torch.allclose(q, 0.5 * raw, atol=1e-6), \
            "curvature_drift_explicit_full should be 0.5 * Tr(Sigma * H)"

    def test_regularized_pseudoinverse_matches_pinv(self):
        """On well-conditioned input, regularized pseudoinverse ~ torch.linalg.pinv."""
        B, D, d = 4, 5, 2
        # Well-conditioned dphi
        dphi = torch.randn(B, D, d) * 2 + 0.5
        pinv_torch = torch.linalg.pinv(dphi)

        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g, eps=1e-8)  # very small eps
        pinv_stable = ginv @ dphi.mT

        assert torch.allclose(pinv_torch, pinv_stable, atol=1e-3), \
            "Regularized pseudoinverse should match torch.linalg.pinv on well-conditioned input"

    def test_phat_idempotency(self, ae):
        """P_hat should be approximately idempotent (P^2 ≈ P)."""
        z = torch.randn(4, 2)
        dphi = ae.decoder.jacobian_network(z)
        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        P_hat = dphi @ ginv @ dphi.mT
        P_hat = 0.5 * (P_hat + P_hat.mT)

        P_hat_sq = P_hat @ P_hat
        error = torch.linalg.matrix_norm(P_hat_sq - P_hat, ord="fro")
        # eps regularization means it won't be perfectly idempotent
        assert (error < 0.1).all(), \
            f"P_hat should be approximately idempotent, max error: {error.max().item():.4f}"

    def test_symmetry_after_computation(self, ae, sample_data):
        """P_hat, Lambda_tan, Sigma_z_obs should be symmetric after computation."""
        x, _, Lambda = sample_data
        z = ae.encoder(x).detach()
        dphi = ae.decoder.jacobian_network(z)

        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi.mT

        P_hat = dphi @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)
        assert torch.allclose(P_hat, P_hat.mT, atol=1e-6), "P_hat should be symmetric"

        Lambda_tan = P_hat @ Lambda @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        assert torch.allclose(Lambda_tan, Lambda_tan.mT, atol=1e-6), \
            "Lambda_tan should be symmetric"

        Sigma_z_obs = pinv @ Lambda_tan @ pinv.mT
        Sigma_z_obs = 0.5 * (Sigma_z_obs + Sigma_z_obs.mT)
        assert torch.allclose(Sigma_z_obs, Sigma_z_obs.mT, atol=1e-6), \
            "Sigma_z_obs should be symmetric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
