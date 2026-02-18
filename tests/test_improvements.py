"""
Tests for the AEML improvements:
- GPU support
- Vectorized Jacobian/Hessian
- Proper weight tying
- Training stability
- Numerical stability
"""
import torch
import torch.nn as nn
import pytest

from src.numeric.ffnn import FeedForwardNeuralNet, TiedWeightParametrization
from src.numeric.autoencoders import AutoEncoder
from src.numeric.training import TrainingConfig, MultiModelTrainer, ModelConfig
from src.numeric.losses import LossWeights
from src.numeric.geometry import (
    curvature_drift_explicit,
    curvature_drift_explicit_full,
    curvature_drift_hessian_free,
    curvature_drift_hessian_free_full,
    orthogonal_projection_from_jacobian,
    ambient_quadratic_variation_drift,
    transform_covariance,
)


class TestDeviceConfiguration:
    """Tests for GPU/device support."""

    def test_training_config_device_default(self):
        """TrainingConfig should have device field with sensible default."""
        config = TrainingConfig()
        assert hasattr(config, 'device')
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected

    def test_trainer_stores_device(self):
        """MultiModelTrainer should store device from config."""
        config = TrainingConfig(device="cpu")
        trainer = MultiModelTrainer(config)
        assert trainer.device == torch.device("cpu")

    def test_model_moved_to_device(self):
        """Model should be moved to configured device."""
        config = TrainingConfig(device="cpu", hidden_dim=8, latent_dim=2, input_dim=3)
        trainer = MultiModelTrainer(config)
        trainer.add_model(ModelConfig(name="test", loss_weights=LossWeights()))

        model = trainer.models["test"]
        # Check first encoder layer is on CPU
        assert next(model.parameters()).device == torch.device("cpu")


class TestVectorizedJacobian:
    """Tests for vectorized Jacobian computation."""

    def setup_method(self):
        """Create a simple network for testing."""
        self.net = FeedForwardNeuralNet(
            neurons=[3, 8, 2],
            activations=[nn.Tanh(), None]
        )
        self.x = torch.randn(5, 3)

    def test_jacobian_vmap_shape(self):
        """Vectorized Jacobian should have correct shape."""
        jac = self.net.jacobian_network(self.x, method="vmap")
        assert jac.shape == (5, 2, 3)  # (batch, output_dim, input_dim)

    def test_jacobian_vmap_matches_autograd(self):
        """Vectorized Jacobian should match autograd result."""
        jac_vmap = self.net.jacobian_network(self.x.clone(), method="vmap")
        jac_autograd = self.net.jacobian_network(self.x.clone(), method="autograd")
        assert torch.allclose(jac_vmap, jac_autograd, rtol=1e-4, atol=1e-6)

    def test_jacobian_default_is_vmap(self):
        """Default method should be vmap."""
        jac_default = self.net.jacobian_network(self.x.clone())
        jac_vmap = self.net.jacobian_network(self.x.clone(), method="vmap")
        assert torch.allclose(jac_default, jac_vmap)


class TestVectorizedHessian:
    """Tests for vectorized Hessian computation."""

    def setup_method(self):
        """Create a simple network for testing."""
        self.net = FeedForwardNeuralNet(
            neurons=[3, 8, 2],
            activations=[nn.Tanh(), None]
        )
        self.x = torch.randn(4, 3)

    def test_hessian_vmap_shape(self):
        """Vectorized Hessian should have correct shape."""
        hess = self.net.hessian_network(self.x, method="vmap")
        assert hess.shape == (4, 2, 3, 3)  # (batch, output_dim, input_dim, input_dim)

    def test_hessian_vmap_matches_autograd(self):
        """Vectorized Hessian should match autograd result."""
        hess_vmap = self.net.hessian_network(self.x.clone(), method="vmap")
        hess_autograd = self.net.hessian_network(self.x.clone(), method="autograd")
        assert torch.allclose(hess_vmap, hess_autograd, rtol=1e-4, atol=1e-5)


class TestPathBasedMethods:
    """Tests for vectorized path-based Jacobian/Hessian."""

    def setup_method(self):
        """Create a simple network for testing."""
        self.net = FeedForwardNeuralNet(
            neurons=[3, 4, 2],
            activations=[nn.Tanh(), None]
        )
        # (num_paths, n_steps, d)
        self.x = torch.randn(2, 5, 3)

    def test_jacobian_for_paths_vmap_shape(self):
        """Vectorized Jacobian for paths should have correct shape."""
        jac = self.net.jacobian_network_for_paths(self.x, method="vmap")
        # (num_paths, n_steps, output_dim, input_dim)
        assert jac.shape == (2, 5, 2, 3)

    def test_jacobian_for_paths_vmap_matches_autograd(self):
        """Vectorized path Jacobian should match autograd result."""
        jac_vmap = self.net.jacobian_network_for_paths(self.x.clone(), method="vmap")
        jac_autograd = self.net.jacobian_network_for_paths(self.x.clone(), method="autograd")
        assert torch.allclose(jac_vmap, jac_autograd, rtol=1e-4, atol=1e-5)

    def test_hessian_for_paths_vmap_shape(self):
        """Vectorized Hessian for paths should have correct shape."""
        hess = self.net.hessian_network_for_paths(self.x, method="vmap")
        # (num_paths, n_steps, output_dim, d, d)
        assert hess.shape == (2, 5, 2, 3, 3)

    def test_hessian_for_paths_vmap_matches_autograd(self):
        """Vectorized path Hessian should match autograd result."""
        # Use smaller input due to computational cost of autograd version
        x_small = torch.randn(2, 3, 3)
        hess_vmap = self.net.hessian_network_for_paths(x_small.clone(), method="vmap")
        hess_autograd = self.net.hessian_network_for_paths(x_small.clone(), method="autograd")
        assert torch.allclose(hess_vmap, hess_autograd, rtol=1e-4, atol=1e-5)


class TestBrownianDrift:
    """Tests for vectorized Brownian drift computation."""

    def setup_method(self):
        """Create an autoencoder for testing."""
        self.ae = AutoEncoder(
            extrinsic_dim=5,
            intrinsic_dim=2,
            hidden_dims=[4],
            encoder_act=nn.Tanh(),
            decoder_act=nn.Tanh(),
            tie_weights=False  # Avoid tied weight complications for this test
        )
        self.z = torch.randn(3, 2)  # (batch, intrinsic_dim)

    def test_brownian_drift_2_vmap_shape(self):
        """Vectorized Brownian drift should have correct shape."""
        drift = self.ae.brownian_drift_2(self.z, method="vmap")
        assert drift.shape == (3, 2)  # (batch, intrinsic_dim)

    def test_brownian_drift_2_vmap_matches_loop(self):
        """Vectorized Brownian drift should match loop-based result."""
        drift_vmap = self.ae.brownian_drift_2(self.z.clone(), method="vmap")
        drift_loop = self.ae.brownian_drift_2(self.z.clone(), method="loop")
        assert torch.allclose(drift_vmap, drift_loop, rtol=1e-4, atol=1e-5)


class TestWeightTying:
    """Tests for proper weight tying with parametrization."""

    def test_tied_weight_parametrization(self):
        """TiedWeightParametrization should return encoder weight transpose."""
        encoder_layer = nn.Linear(3, 5)
        param = TiedWeightParametrization(encoder_layer)

        # Should return transpose of encoder weight
        result = param(torch.zeros(5, 3))  # Ignored input
        assert torch.allclose(result, encoder_layer.weight.t())

    def test_autoencoder_weight_tying(self):
        """AutoEncoder with tied weights should maintain tie during forward pass."""
        ae = AutoEncoder(
            extrinsic_dim=10,
            intrinsic_dim=3,
            hidden_dims=[5],
            encoder_act=nn.Tanh(),
            decoder_act=nn.Tanh(),
            tie_weights=True
        )

        # Get first encoder and last decoder layers
        encoder_first = ae.encoder.layers[0]
        decoder_last = ae.decoder.layers[-1]

        # Weights should be transposes
        encoder_weight = encoder_first.weight
        decoder_weight = decoder_last.weight

        assert torch.allclose(decoder_weight, encoder_weight.t())

    def test_weight_tying_persists_after_forward(self):
        """Tied weights should remain tied after forward pass."""
        ae = AutoEncoder(
            extrinsic_dim=5,
            intrinsic_dim=2,
            hidden_dims=[3],
            encoder_act=nn.Tanh(),
            decoder_act=nn.Tanh(),
            tie_weights=True
        )

        x = torch.randn(10, 5)
        _ = ae(x)  # Forward pass

        # Check weights still tied
        encoder_first = ae.encoder.layers[0]
        decoder_last = ae.decoder.layers[-1]

        assert torch.allclose(decoder_last.weight, encoder_first.weight.t())


class TestTrainingStability:
    """Tests for training stability features."""

    def test_grad_clip_config(self):
        """TrainingConfig should have grad_clip_max_norm field."""
        config = TrainingConfig()
        assert hasattr(config, 'grad_clip_max_norm')
        assert config.grad_clip_max_norm == 1.0

    def test_scheduler_created(self):
        """MultiModelTrainer should create learning rate schedulers."""
        config = TrainingConfig(device="cpu")
        trainer = MultiModelTrainer(config)
        trainer.add_model(ModelConfig(name="test", loss_weights=LossWeights()))

        assert "test" in trainer.schedulers
        assert isinstance(
            trainer.schedulers["test"],
            torch.optim.lr_scheduler.ReduceLROnPlateau
        )


class TestNumericalStability:
    """Tests for numerical stability improvements."""

    def test_orthogonal_projection_with_eps(self):
        """orthogonal_projection_from_jacobian should accept eps parameter."""
        jacobian = torch.randn(3, 5, 2)  # (batch, output_dim, input_dim)

        # Should not raise with default eps
        proj = orthogonal_projection_from_jacobian(jacobian)
        assert proj.shape == (3, 5, 5)

        # Should work with custom eps
        proj_eps = orthogonal_projection_from_jacobian(jacobian, eps=1e-4)
        assert proj_eps.shape == (3, 5, 5)

    def test_singular_jacobian_handling(self):
        """Should handle near-singular Jacobians gracefully."""
        # Create near-singular Jacobian
        jacobian = torch.zeros(2, 3, 2)
        jacobian[0, 0, 0] = 1.0
        jacobian[0, 1, 1] = 1e-8  # Near zero
        jacobian[1, 0, 0] = 1.0
        jacobian[1, 1, 1] = 1.0

        # Should not raise due to eps regularization
        proj = orthogonal_projection_from_jacobian(jacobian, eps=1e-6)
        assert not torch.isnan(proj).any()
        assert not torch.isinf(proj).any()


class TestFullSpaceCurvature:
    """Tests for full-space (no normal projection) curvature functions."""

    def setup_method(self):
        B, D, d = 4, 5, 2
        self.B, self.D, self.d = B, D, d
        self.decoder_hessian = torch.randn(B, D, d, d)
        # Make local_cov symmetric positive definite
        L = torch.randn(B, d, d)
        self.local_cov = torch.bmm(L, L.mT) + 0.1 * torch.eye(d).unsqueeze(0)
        # Projection matrix
        dphi = torch.randn(B, D, d)
        self.normal_proj = torch.eye(D).unsqueeze(0) - orthogonal_projection_from_jacobian(dphi)

    def test_curvature_drift_explicit_full_shape(self):
        result = curvature_drift_explicit_full(self.decoder_hessian, self.local_cov)
        assert result.shape == (self.B, self.D)

    def test_curvature_drift_explicit_full_no_projection(self):
        """Full version should equal explicit version WITHOUT the normal projection step."""
        full = curvature_drift_explicit_full(self.decoder_hessian, self.local_cov)
        expected = 0.5 * ambient_quadratic_variation_drift(self.local_cov, self.decoder_hessian)
        assert torch.allclose(full, expected, atol=1e-6)

    def test_curvature_drift_explicit_full_vs_projected(self):
        """Full version should differ from projected version (unless normal_proj is identity)."""
        full = curvature_drift_explicit_full(self.decoder_hessian, self.local_cov)
        projected = curvature_drift_explicit(self.decoder_hessian, self.local_cov, self.normal_proj)
        # They should generally NOT be equal
        assert not torch.allclose(full, projected, atol=1e-4)

    def test_curvature_drift_hessian_free_full_shape(self):
        ae = AutoEncoder(
            extrinsic_dim=self.D, intrinsic_dim=self.d,
            hidden_dims=[8], encoder_act=nn.Tanh(), decoder_act=nn.Tanh(),
            tie_weights=False,
        )
        z = torch.randn(self.B, self.d)
        result = curvature_drift_hessian_free_full(ae.decoder, z, self.local_cov)
        assert result.shape == (self.B, self.D)

    def test_curvature_drift_hessian_free_full_matches_explicit(self):
        """Hessian-free full should match explicit full for same decoder."""
        ae = AutoEncoder(
            extrinsic_dim=self.D, intrinsic_dim=self.d,
            hidden_dims=[8], encoder_act=nn.Tanh(), decoder_act=nn.Tanh(),
            tie_weights=False,
        )
        z = torch.randn(self.B, self.d)
        d2phi = ae.hessian_decoder(z)

        explicit = curvature_drift_explicit_full(d2phi, self.local_cov)
        hfree = curvature_drift_hessian_free_full(ae.decoder, z, self.local_cov)
        assert torch.allclose(explicit, hfree, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
