"""
Multi-stage SDE training pipeline.

Stage 1: Train autoencoder with recon + T + K (existing MultiModelTrainer).
Stage 2: Freeze AE, train drift_net with tangential drift matching.
Stage 3: Freeze AE, train diffusion_net with tangent-projected cov + K reg.

Conventions:
- In Stages 2/3: freeze entire autoencoder AND detach z.
- Do NOT wrap in torch.no_grad() — torch.func transforms need the graph.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .sde_losses import tangential_drift_loss, tangent_diffusion_loss


class SDEPipelineTrainer:
    """Coordinates the 3-stage data-driven latent SDE training pipeline."""

    def __init__(self, autoencoder, drift_net, diffusion_net, device="cpu"):
        self.device = torch.device(device)
        self.autoencoder = autoencoder.to(self.device)
        self.drift_net = drift_net.to(self.device)
        self.diffusion_net = diffusion_net.to(self.device)

    def _make_sde_dataloader(self, x, v, Lambda, batch_size):
        """Minimal dataloader from raw tensors: (x, v, Lambda)."""
        dataset = TensorDataset(x, v, Lambda)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _freeze_autoencoder(self):
        """Freeze entire autoencoder for Stages 2/3."""
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad_(False)

    def _unfreeze_autoencoder(self):
        """Unfreeze autoencoder (restore after Stages 2/3 if needed)."""
        self.autoencoder.train()
        for p in self.autoencoder.parameters():
            p.requires_grad_(True)

    def train_stage2(
        self, x, v, Lambda, epochs, lr=1e-3, batch_size=32, print_interval=100,
    ):
        """
        Stage 2: Train drift_net with tangential drift matching (frozen AE).

        Args:
            x: Ambient samples, shape (N, D).
            v: Ambient drift/velocity, shape (N, D).
            Lambda: Ambient covariance, shape (N, D, D).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            print_interval: Print loss every N epochs.

        Returns:
            List of epoch-averaged losses.
        """
        self._freeze_autoencoder()
        self.drift_net.train()
        optimizer = torch.optim.Adam(self.drift_net.parameters(), lr=lr)
        loader = self._make_sde_dataloader(x, v, Lambda, batch_size)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_b, v_b, Lambda_b in loader:
                x_b = x_b.to(self.device)
                v_b = v_b.to(self.device)
                Lambda_b = Lambda_b.to(self.device)
                z = self.autoencoder.encoder(x_b).detach()
                loss = tangential_drift_loss(
                    self.autoencoder.decoder, self.drift_net, z, v_b, Lambda_b,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 2 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        return losses

    def train_stage3(
        self, x, v, Lambda, epochs, lr=1e-3, batch_size=32,
        lambda_K=0.1, print_interval=100,
    ):
        """
        Stage 3: Train diffusion_net with tangent-projected cov + K reg (frozen AE).

        Args:
            x: Ambient samples, shape (N, D).
            v: Ambient drift/velocity, shape (N, D).
            Lambda: Ambient covariance, shape (N, D, D).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            lambda_K: Weight for K identity regularization.
            print_interval: Print loss every N epochs.

        Returns:
            List of epoch-averaged losses.
        """
        self._freeze_autoencoder()
        self.diffusion_net.train()
        optimizer = torch.optim.Adam(self.diffusion_net.parameters(), lr=lr)
        loader = self._make_sde_dataloader(x, v, Lambda, batch_size)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_b, v_b, Lambda_b in loader:
                x_b = x_b.to(self.device)
                v_b = v_b.to(self.device)
                Lambda_b = Lambda_b.to(self.device)
                z = self.autoencoder.encoder(x_b).detach()
                loss = tangent_diffusion_loss(
                    self.autoencoder.decoder, self.diffusion_net,
                    z, v_b, Lambda_b, lambda_K=lambda_K,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 3 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        return losses

    @torch.no_grad()
    def simulate(self, z0, n_steps, dt, dW=None):
        """
        Euler-Maruyama simulation in latent space using learned nets.

        dz = drift_net(z) * dt + diffusion_net(z) @ dW

        Args:
            z0: Initial latent points, shape (B, d).
            n_steps: Number of time steps.
            dt: Time step size.
            dW: Optional pre-generated Brownian increments, shape (B, n_steps, d).
                If None, generates standard normal increments.

        Returns:
            z_traj: Latent trajectory, shape (B, n_steps+1, d).
            x_traj: Ambient trajectory, shape (B, n_steps+1, D).
        """
        self.drift_net.eval()
        self.diffusion_net.eval()
        B, d = z0.shape
        device = z0.device

        if dW is None:
            dW = torch.randn(B, n_steps, d, device=device) * (dt ** 0.5)
        else:
            # dW should already be scaled by sqrt(dt) or be raw N(0,1)
            # Convention: dW are raw N(0,1), we scale by sqrt(dt)
            dW = dW * (dt ** 0.5)

        z_traj = torch.zeros(B, n_steps + 1, d, device=device)
        z_traj[:, 0] = z0
        z = z0.clone()

        for t in range(n_steps):
            b_z = self.drift_net(z)        # (B, d)
            sigma_z = self.diffusion_net(z)  # (B, d, d)
            noise = (sigma_z @ dW[:, t].unsqueeze(-1)).squeeze(-1)  # (B, d)
            z = z + b_z * dt + noise
            z_traj[:, t + 1] = z

        # Decode all at once
        z_flat = z_traj.reshape(B * (n_steps + 1), d)
        x_flat = self.autoencoder.decoder(z_flat)
        D = x_flat.shape[-1]
        x_traj = x_flat.reshape(B, n_steps + 1, D)

        return z_traj, x_traj
