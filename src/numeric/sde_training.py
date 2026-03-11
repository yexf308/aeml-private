"""
Multi-stage SDE training pipeline.

Stage 1: Train autoencoder with recon + T + K (existing MultiModelTrainer).
Stage 2: Freeze AE, train drift_net with tangential drift matching.
Stage 3: Freeze AE, train diffusion_net with ambient covariance matching.

Conventions:
- In Stages 2/3: freeze entire autoencoder AND detach z.
- Do NOT wrap in torch.no_grad() — torch.func transforms need the graph.
"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .sde_losses import (
    tangential_drift_loss, ambient_diffusion_loss, latent_diffusion_loss,
    drift_smoothness_loss, diffusion_smoothness_loss,
    latent_drift_regression_loss, encoder_pullback_drift_loss,
)


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

    def precompute_decoder_derivatives(self, x, batch_size=64):
        """Precompute z, dphi, d2phi for all training points (frozen AE).

        This avoids recomputing expensive Hessians every batch in Stages 2/3.

        Args:
            x: Ambient samples, shape (N, D).
            batch_size: Batch size for precomputation.

        Returns:
            z: Latent encodings, shape (N, d). Detached.
            dphi: Decoder Jacobians, shape (N, D, d). Detached.
            d2phi: Decoder Hessians, shape (N, D, d, d). Detached.
        """
        self._freeze_autoencoder()
        z_all, dphi_all, d2phi_all = [], [], []
        for i in range(0, len(x), batch_size):
            x_b = x[i:i + batch_size].to(self.device)
            z = self.autoencoder.encoder(x_b).detach()
            dphi = self.autoencoder.decoder.jacobian_network(z).detach()
            d2phi = self.autoencoder.decoder.hessian_network(z).detach()
            z_all.append(z)
            dphi_all.append(dphi)
            d2phi_all.append(d2phi)
        return torch.cat(z_all), torch.cat(dphi_all), torch.cat(d2phi_all)

    def train_stage2(
        self, x, v, Lambda, epochs, lr=1e-3, batch_size=32, print_interval=100,
        lambda_smooth=0.0, aug_sigma=0.1, use_metric=True,
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
            lambda_smooth: Weight for drift smoothness regularization (0 = off).
            aug_sigma: Std of Gaussian noise for augmented latent points.
            use_metric: If True, metric-weighted smoothness; if False, Euclidean.

        Returns:
            List of epoch-averaged losses.
        """
        self._freeze_autoencoder()
        self.drift_net.train()
        optimizer = torch.optim.Adam(self.drift_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        loader = self._make_sde_dataloader(x, v, Lambda, batch_size)
        losses = []
        best_loss, best_state = float("inf"), None

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

                if lambda_smooth > 0.0:
                    z_aug = z + torch.randn_like(z) * aug_sigma
                    loss = loss + lambda_smooth * drift_smoothness_loss(
                        self.autoencoder.decoder, self.drift_net, z_aug,
                        use_metric=use_metric,
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.drift_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.drift_net.state_dict())
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 2 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        self.drift_net.load_state_dict(best_state)
        return losses

    def train_stage2_precomputed(
        self, z, dphi, d2phi, v, Lambda, epochs, lr=1e-3,
        batch_size=32, print_interval=100,
        lambda_smooth=0.0, aug_sigma=0.1, use_metric=True,
        q_override=None,
    ):
        """Stage 2 with precomputed decoder derivatives (much faster).

        Note: the smoothness loss recomputes dphi at augmented points via the
        decoder, since precomputed derivatives are only valid at training z.

        Args:
            q_override: Optional precomputed curvature correction, shape (N, D).
                If provided, used instead of computing q from d2phi.
        """
        self.drift_net.train()
        optimizer = torch.optim.Adam(self.drift_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        if q_override is not None:
            dataset = TensorDataset(z, dphi, d2phi, v, Lambda, q_override)
        else:
            dataset = TensorDataset(z, dphi, d2phi, v, Lambda)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        best_loss, best_state = float("inf"), None

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if q_override is not None:
                    z_b, dphi_b, d2phi_b, v_b, Lambda_b, q_b = batch
                    loss = tangential_drift_loss(
                        self.autoencoder.decoder, self.drift_net,
                        z_b, v_b, Lambda_b, dphi=dphi_b, d2phi=d2phi_b,
                        q_override=q_b,
                    )
                else:
                    z_b, dphi_b, d2phi_b, v_b, Lambda_b = batch
                    loss = tangential_drift_loss(
                        self.autoencoder.decoder, self.drift_net,
                        z_b, v_b, Lambda_b, dphi=dphi_b, d2phi=d2phi_b,
                    )

                if lambda_smooth > 0.0:
                    z_aug = z_b + torch.randn_like(z_b) * aug_sigma
                    loss = loss + lambda_smooth * drift_smoothness_loss(
                        self.autoencoder.decoder, self.drift_net, z_aug,
                        use_metric=use_metric,
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.drift_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.drift_net.state_dict())
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 2 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        self.drift_net.load_state_dict(best_state)
        return losses

    def precompute_encoder_jacobian(self, x, batch_size=64):
        """Precompute encoder Jacobian for all training points.

        Args:
            x: Ambient samples, shape (N, D).
            batch_size: Batch size for computation.

        Returns:
            dpi: Encoder Jacobian, shape (N, d, D). Detached.
        """
        import torch
        self._freeze_autoencoder()
        dpi_all = []
        for i in range(0, len(x), batch_size):
            x_b = x[i:i + batch_size].to(self.device)
            dpi = torch.func.vmap(torch.func.jacrev(self.autoencoder.encoder))(x_b).detach()
            dpi_all.append(dpi)
        return torch.cat(dpi_all)

    def _ito_correction_full_hessian(self, x, Lambda, batch_size=8):
        """Compute ½⟨Λ, ∇²π⟩ by forming the full encoder Hessian.

        Cost: O(d·D²) per sample. Use for small D.
        """
        N = x.shape[0]
        corrections = []
        for i in range(0, N, batch_size):
            x_b = x[i:i + batch_size].to(self.device)
            Lambda_b = Lambda[i:i + batch_size].to(self.device)
            Lambda_b = 0.5 * (Lambda_b + Lambda_b.mT)
            d2pi_b = self.autoencoder.encoder_hessian(x_b).detach()  # (B, d, D, D)
            corr_b = 0.5 * torch.einsum('brs, birs -> bi', Lambda_b, d2pi_b)
            corrections.append(corr_b)
        return torch.cat(corrections)

    def _ito_correction_hvp(self, x, Lambda, batch_size=8):
        """Compute ½⟨Λ, ∇²π⟩ via Hessian-vector products.

        Factors Λ = Σ_m λ_m u_m u_m^T (spectral decomposition, rank r ≤ d)
        and evaluates u_m^T ∇²π u_m via forward-over-reverse autodiff,
        avoiding the full (d, D, D) Hessian.

        Cost: O(d·r) autodiff passes per sample vs O(d·D²) for full Hessian.
        Speedup factor: ~D/r ≈ D/d.
        """
        from torch.func import jacrev, jvp, vmap

        N, D = x.shape
        encoder = self.autoencoder.encoder
        d = encoder.output_dim

        def enc_fn(x_single):
            return encoder(x_single.unsqueeze(0)).squeeze(0)

        def second_dir_deriv(x_single, u_single):
            """u^T ∇²π(x) u — second directional derivative, shape (d,)."""
            jac_fn = jacrev(enc_fn)
            _, H_u = jvp(jac_fn, (x_single,), (u_single,))
            # H_u: (d, D) — row j is ∇²π^j · u
            return (H_u * u_single).sum(-1)  # (d,)

        batched_sdd = vmap(second_dir_deriv)

        corrections = []
        for i in range(0, N, batch_size):
            x_b = x[i:i + batch_size].to(self.device)
            Lambda_b = Lambda[i:i + batch_size].to(self.device)
            Lambda_b = 0.5 * (Lambda_b + Lambda_b.mT)
            B = x_b.shape[0]

            # Spectral decomposition: eigenvalues ascending
            eigvals, eigvecs = torch.linalg.eigh(Lambda_b)

            # Keep top r components (rank Λ ≤ d; take d+2 for safety)
            r = min(D, d + 2)
            top_vals = eigvals[:, -r:]       # (B, r)
            top_vecs = eigvecs[:, :, -r:]    # (B, D, r)

            # Zero out negligible eigenvalues
            threshold = 1e-6 * top_vals[:, -1:].clamp(min=1e-12)
            mask = (top_vals > threshold).float()
            top_vals = top_vals * mask

            corr_b = torch.zeros(B, d, device=self.device)
            for m in range(r):
                lam_m = top_vals[:, m]       # (B,)
                if lam_m.max() < 1e-12:
                    continue
                u_m = top_vecs[:, :, m]      # (B, D)
                sdd = batched_sdd(x_b, u_m)  # (B, d)
                corr_b = corr_b + lam_m.unsqueeze(-1) * sdd

            corrections.append(0.5 * corr_b.detach())

        return torch.cat(corrections)

    def precompute_enc_pull_target(self, x, v, Lambda, dphi, batch_size=8):
        """Precompute encoder-pullback drift target: b_z = Dπ·v + ½Λ:∇²π.

        This is the exact Itô drift for the learned encoder, computed via
        autodiff. The result is a fixed regression target for Stage 2.

        For large D, uses Hessian-vector products (HVP) to avoid forming the
        full (d, D, D) encoder Hessian. The HVP method exploits the low rank
        of Λ (rank ≤ d) for O(d²) autodiff passes instead of O(d·D²).

        Args:
            x: Ambient samples, shape (N, D).
            v: Ambient drift, shape (N, D).
            Lambda: Ambient covariance, shape (N, D, D).
            dphi: Decoder Jacobian, shape (N, D, d). For metric computation.
            batch_size: Batch size for encoder Hessian computation.

        Returns:
            b_z_target: Encoder-pullback drift target, shape (N, d).
            g: Metric tensor, shape (N, d, d).
        """
        import torch
        self._freeze_autoencoder()
        N, D = x.shape
        d = dphi.shape[-1]

        # Encoder Jacobian
        dpi = self.precompute_encoder_jacobian(x, batch_size=batch_size * 4)

        # First-order term: Dπ·v
        b_z_1st = (dpi @ v.unsqueeze(-1)).squeeze(-1)  # (N, d)

        # Second-order term: ½Λ:∇²π
        # Use HVP when D >> d (avoids O(d·D²) full Hessian)
        if D > 4 * d:
            ito_correction = self._ito_correction_hvp(x, Lambda, batch_size)
        else:
            ito_correction = self._ito_correction_full_hessian(x, Lambda, batch_size)

        b_z_target = (b_z_1st + ito_correction).detach()
        g = (dphi.mT @ dphi).detach()

        return b_z_target, g

    def train_stage2_encoder_pullback(
        self, x, z, dphi, d2phi, dpi, v, Lambda, epochs, lr=1e-3,
        batch_size=32, print_interval=100,
        lambda_smooth=0.0, aug_sigma=0.1, use_metric=True,
    ):
        """Stage 2 with D²(π∘φ)=0 identity (enc_pull_v2, deprecated).

        Uses b_z_target = Dπ·(v - q_φ) via the identity D²(π∘φ) = 0.
        NOTE: This identity doesn't hold for learned AEs (||D²(π∘φ)|| ~ 1).
        Prefer precompute_enc_pull_target + train_stage2_regression.

        Args:
            x: Ambient samples, shape (N, D).
            z: Latent encodings, shape (N, d).
            dphi: Decoder Jacobian, shape (N, D, d).
            d2phi: Decoder Hessian, shape (N, D, d, d).
            dpi: Encoder Jacobian, shape (N, d, D).
            v: Ambient drift, shape (N, D).
            Lambda: Ambient covariance, shape (N, D, D).
        """
        self.drift_net.train()
        optimizer = torch.optim.Adam(self.drift_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        dataset = TensorDataset(x, z, dphi, d2phi, dpi, v, Lambda)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        best_loss, best_state = float("inf"), None

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_b, z_b, dphi_b, d2phi_b, dpi_b, v_b, Lambda_b in loader:
                loss = encoder_pullback_drift_loss(
                    self.autoencoder.encoder, self.autoencoder.decoder,
                    self.drift_net, z_b, x_b, v_b, Lambda_b,
                    dphi=dphi_b, d2phi=d2phi_b, dpi=dpi_b,
                )

                if lambda_smooth > 0.0:
                    z_aug = z_b + torch.randn_like(z_b) * aug_sigma
                    loss = loss + lambda_smooth * drift_smoothness_loss(
                        self.autoencoder.decoder, self.drift_net, z_aug,
                        use_metric=use_metric,
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.drift_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.drift_net.state_dict())
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 2 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        self.drift_net.load_state_dict(best_state)
        return losses

    def train_stage2_regression(
        self, z, b_z_target, g, epochs, lr=1e-3,
        batch_size=32, print_interval=100,
        lambda_smooth=0.0, aug_sigma=0.1, use_metric=True,
    ):
        """Stage 2 with precomputed latent drift target (regression).

        Trains drift_net to match a precomputed b_z_target using
        metric-weighted MSE: (b_z - b_target)^T g (b_z - b_target).

        Used for encoder-pullback and direct-regression oracle conditions.

        Args:
            z: Latent points, shape (N, d).
            b_z_target: Target latent drift, shape (N, d).
            g: Metric tensor, shape (N, d, d).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            print_interval: Print interval.
            lambda_smooth: Drift smoothness weight.
            aug_sigma: Augmentation noise std.
            use_metric: Metric-weighted smoothness.
        """
        self.drift_net.train()
        optimizer = torch.optim.Adam(self.drift_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        dataset = TensorDataset(z, b_z_target, g)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        best_loss, best_state = float("inf"), None

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for z_b, target_b, g_b in loader:
                loss = latent_drift_regression_loss(
                    self.drift_net, z_b, target_b, g=g_b,
                    ambient_dim=self.autoencoder.extrinsic_dim,
                )

                if lambda_smooth > 0.0:
                    z_aug = z_b + torch.randn_like(z_b) * aug_sigma
                    loss = loss + lambda_smooth * drift_smoothness_loss(
                        self.autoencoder.decoder, self.drift_net, z_aug,
                        use_metric=use_metric,
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.drift_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.drift_net.state_dict())
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 2 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        self.drift_net.load_state_dict(best_state)
        return losses

    def train_stage3(
        self, x, Lambda, epochs, lr=1e-3, batch_size=32, print_interval=100,
        lambda_smooth_diff=0.0, aug_sigma=0.1, use_latent_loss=False,
    ):
        """
        Stage 3: Train diffusion_net with ambient covariance matching (frozen AE).

        Args:
            x: Ambient samples, shape (N, D).
            Lambda: Ambient covariance, shape (N, D, D).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            print_interval: Print loss every N epochs.

        Returns:
            List of epoch-averaged losses.
        """
        self._freeze_autoencoder()
        self.diffusion_net.train()
        optimizer = torch.optim.Adam(self.diffusion_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        dataset = TensorDataset(x, Lambda)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        best_loss, best_state = float("inf"), None

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_b, Lambda_b in loader:
                x_b = x_b.to(self.device)
                Lambda_b = Lambda_b.to(self.device)
                z = self.autoencoder.encoder(x_b).detach()
                if use_latent_loss:
                    dphi = self.autoencoder.decoder.jacobian_network(z).detach()
                    loss = latent_diffusion_loss(
                        self.diffusion_net, z, Lambda_b, dphi,
                    )
                else:
                    loss = ambient_diffusion_loss(
                        self.diffusion_net, z, Lambda_b,
                        decoder=self.autoencoder.decoder,
                    )
                if lambda_smooth_diff > 0.0:
                    z_aug = z + torch.randn_like(z) * aug_sigma
                    loss = loss + lambda_smooth_diff * diffusion_smoothness_loss(
                        self.autoencoder.decoder, self.diffusion_net, z_aug,
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.diffusion_net.state_dict())
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 3 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        self.diffusion_net.load_state_dict(best_state)
        return losses

    def train_stage3_precomputed(
        self, z, dphi, Lambda, epochs, lr=1e-3,
        batch_size=32, print_interval=100,
        v=None, d2phi=None, lambda_K=0.0,
        use_latent_loss=False,
    ):
        """Stage 3 with precomputed decoder Jacobians (much faster).

        Args:
            v: Ambient drift, shape (N, D). Required if lambda_K > 0.
            d2phi: Decoder Hessians, shape (N, D, d, d). Required if lambda_K > 0.
            lambda_K: Weight for K identity regularization in diffusion loss.
        """
        self.diffusion_net.train()
        optimizer = torch.optim.Adam(self.diffusion_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        if lambda_K > 0:
            dataset = TensorDataset(z, dphi, d2phi, v, Lambda)
        else:
            dataset = TensorDataset(z, dphi, Lambda)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        best_loss, best_state = float("inf"), None

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if lambda_K > 0:
                    z_b, dphi_b, d2phi_b, v_b, Lambda_b = batch
                    loss = ambient_diffusion_loss(
                        self.diffusion_net, z_b, Lambda_b, dphi=dphi_b,
                        v=v_b, d2phi=d2phi_b, lambda_K=lambda_K,
                    )
                elif use_latent_loss:
                    z_b, dphi_b, Lambda_b = batch
                    loss = latent_diffusion_loss(
                        self.diffusion_net, z_b, Lambda_b, dphi=dphi_b,
                    )
                else:
                    z_b, dphi_b, Lambda_b = batch
                    loss = ambient_diffusion_loss(
                        self.diffusion_net, z_b, Lambda_b, dphi=dphi_b,
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.diffusion_net.state_dict())
            if print_interval and (epoch + 1) % print_interval == 0:
                print(f"  Stage 3 epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        self.diffusion_net.load_state_dict(best_state)
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
