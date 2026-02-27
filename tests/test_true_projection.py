"""Tests for the use_true_projection flag in autoencoder_loss."""
import torch
import torch.nn as nn
import pytest
from src.numeric.autoencoders import AutoEncoder
from src.numeric.losses import autoencoder_loss, LossWeights
from src.numeric.training import ModelConfig


@pytest.fixture
def setup():
    """Small AE with synthetic data where P̂ ≠ P★."""
    torch.manual_seed(0)
    ae = AutoEncoder(
        extrinsic_dim=3, intrinsic_dim=2, hidden_dims=[8],
        encoder_act=nn.Tanh(), decoder_act=nn.Tanh(),
    )
    B = 4
    x = torch.randn(B, 3)
    mu = torch.randn(B, 3)
    cov = torch.eye(3).unsqueeze(0).expand(B, -1, -1) * 0.1
    # True projection: rank-2 projector onto first two axes
    p = torch.zeros(B, 3, 3)
    p[:, 0, 0] = 1.0
    p[:, 1, 1] = 1.0
    targets = (x, mu, cov, p)
    lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1)
    return ae, targets, lw


def test_use_true_projection_differs(setup):
    """Loss with true projection should differ from learned projection."""
    ae, targets, lw = setup
    loss_learned = autoencoder_loss(ae, targets, lw, use_true_projection=False)
    loss_true = autoencoder_loss(ae, targets, lw, use_true_projection=True)
    assert not torch.allclose(loss_learned, loss_true), \
        "Losses should differ when P̂ ≠ P★"


def test_use_true_projection_default_is_learned(setup):
    """Default behavior (no flag) should match use_true_projection=False."""
    ae, targets, lw = setup
    loss_default = autoencoder_loss(ae, targets, lw)
    loss_explicit = autoencoder_loss(ae, targets, lw, use_true_projection=False)
    assert torch.allclose(loss_default, loss_explicit)


def test_use_true_projection_no_curvature_is_noop(setup):
    """When curvature weight is 0, the flag should have no effect."""
    ae, targets, lw_no_k = setup
    lw_no_k = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.0)
    loss_learned = autoencoder_loss(ae, targets, lw_no_k, use_true_projection=False)
    loss_true = autoencoder_loss(ae, targets, lw_no_k, use_true_projection=True)
    assert torch.allclose(loss_learned, loss_true)


def test_model_config_use_true_projection():
    """ModelConfig should accept and store use_true_projection."""
    mc = ModelConfig(
        name="test",
        loss_weights=LossWeights(curvature=0.1),
        use_true_projection=True,
    )
    assert mc.use_true_projection is True

    mc_default = ModelConfig(name="test2", loss_weights=LossWeights())
    assert mc_default.use_true_projection is False
