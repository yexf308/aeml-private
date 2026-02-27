from .autoencoders import AutoEncoder
from .losses import autoencoder_loss, LossWeights
from .datasets import DatasetBatch
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field

# TODO: TrainingConfig holds a latent and hidden dim size. Why should it? ModelConfig handles that.
# We aren't necessarily testing the same architectures. So it's not really a training hyperparameter.
#
# If you look below you will notice 'add_model' pulls from it for default architecture if not specified in
# the current ModelConfig instance.
def _default_device() -> str:
    """Return default device based on CUDA availability."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters"""
    epochs: int = 1000
    n_samples: int = 100
    input_dim: int = 3
    hidden_dim: int = 32
    latent_dim: int = 2
    learning_rate: float = 0.0001
    batch_size: int = 20
    test_size: float = 0.985
    print_interval: int = 100
    embed: bool = False
    embedding_dim: int = 20
    embedding_seed: int = 17
    effective_dim: int = 0
    device: str = field(default_factory=_default_device)
    grad_clip_max_norm: float = 1.0


@dataclass
class TrainingPhase:
    """A training phase with specific loss weights and duration."""
    epochs: int
    loss_weights: LossWeights
    name: str = ""  # Optional name for logging


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    loss_weights: LossWeights
    architecture_params: Optional[Dict[str, Any]] = None
    # Optional training schedule: list of phases to execute in order
    # If provided, loss_weights is used as the final phase weights
    training_schedule: Optional[list] = None  # List[TrainingPhase]
    use_true_projection: bool = False

class MultiModelTrainer:
    """Class to handle training multiple models simultaneously"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.models: Dict[str, AutoEncoder] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._has_local_cov = False

    def add_model(self, model_config: ModelConfig):
        """Add a model to the training pipeline"""
        # Create model with architecture params or default config
        effective_dim = self.config.effective_dim or self.config.input_dim
        arch_params = model_config.architecture_params or {
            "extrinsic_dim": effective_dim,
            "intrinsic_dim": self.config.latent_dim,
            "hidden_dims": [self.config.hidden_dim],
            "encoder_act": nn.Tanh(),
            "decoder_act": nn.Tanh(),
        }

        model = AutoEncoder(**arch_params)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100
        )

        self.models[model_config.name] = model
        self.optimizers[model_config.name] = optimizer
        self.schedulers[model_config.name] = scheduler
        self.model_configs[model_config.name] = model_config
    
    def create_data_loader(self, dataset: Union[DatasetBatch, Tuple[torch.Tensor, ...]]):
        """Create a DataLoader for batched training.

        If any model uses tangent_bundle penalty, precompute tangent basis for efficiency.
        """
        # Check if any model can benefit from efficient tangent loss
        needs_tangent_basis = any(
            cfg.loss_weights.tangent_bundle > 0
            for cfg in self.model_configs.values()
        )

        if isinstance(dataset, DatasetBatch):
            self._has_local_cov = dataset.local_cov is not None

            # Precompute tangent basis if needed
            if needs_tangent_basis and dataset.tangent_basis is None:
                dataset.compute_tangent_basis(self.config.latent_dim)

            # Include tangent_basis in the tensor dataset if computed
            if dataset.tangent_basis is not None:
                tensors = dataset.as_tuple() + (dataset.tangent_basis,)
            else:
                tensors = dataset.as_tuple()
        else:
            # Tuple input: disambiguate local_cov (d,d) from tangent_basis (D,d).
            # If 8th element is square (last two dims equal), it's local_cov.
            # tangent_basis is (B, D, d) where D > d, so not square.
            if len(dataset) >= 8 and dataset[7].shape[-1] == dataset[7].shape[-2]:
                self._has_local_cov = True
            else:
                self._has_local_cov = False
            tensors = dataset

        tensor_dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
    
    def train_epoch(self, data_loader, loss_weights_override: Optional[Dict[str, LossWeights]] = None):
        """Train all models for one epoch.

        Args:
            data_loader: DataLoader for training data
            loss_weights_override: Optional dict mapping model names to LossWeights
                                   to use instead of the default config weights.
                                   Useful for training schedules with warm-up phases.
        """
        epoch_losses = {}

        for model_name, model in self.models.items():
            model.train()
            epoch_losses[model_name] = 0.0

        for batch_idx, batch in enumerate(data_loader):
            x, _, mu, cov, p, _, hessians_batch = batch[:7]
            idx = 7
            if self._has_local_cov:
                local_cov_true_batch = batch[idx]
                idx += 1
            else:
                local_cov_true_batch = None
            tangent_basis = batch[idx] if len(batch) > idx else None

            # Move tensors to device
            x = x.to(self.device)
            mu = mu.to(self.device)
            cov = cov.to(self.device)
            p = p.to(self.device)
            hessians_batch = hessians_batch.to(self.device)
            if local_cov_true_batch is not None:
                local_cov_true_batch = local_cov_true_batch.to(self.device)
            if tangent_basis is not None:
                tangent_basis = tangent_basis.to(self.device)
            targets = (x, mu, cov, p)

            for model_name, model in self.models.items():
                optimizer = self.optimizers[model_name]
                # Use override weights if provided, otherwise use config weights
                if loss_weights_override and model_name in loss_weights_override:
                    loss_weights = loss_weights_override[model_name]
                else:
                    loss_weights = self.model_configs[model_name].loss_weights

                optimizer.zero_grad()
                loss = autoencoder_loss(
                    model, targets, loss_weights,
                    tangent_basis=tangent_basis,
                    hessians=hessians_batch,
                    local_cov_true=local_cov_true_batch,
                    use_true_projection=self.model_configs[model_name].use_true_projection,
                )
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.grad_clip_max_norm
                )
                optimizer.step()

                epoch_losses[model_name] += loss.item()

        # Average losses over batches
        num_batches = len(data_loader)
        for model_name in epoch_losses:
            epoch_losses[model_name] /= num_batches
            # Update learning rate scheduler
            self.schedulers[model_name].step(epoch_losses[model_name])

        return epoch_losses

    def train_with_schedule(self, data_loader, model_name: str, schedule: list, print_interval: int = 100):
        """Train a model with a multi-phase schedule.

        Args:
            data_loader: DataLoader for training data
            model_name: Name of the model to train
            schedule: List of TrainingPhase objects defining the training schedule
            print_interval: How often to print progress

        Example schedule for diffeo warm-up:
            schedule = [
                TrainingPhase(epochs=100, loss_weights=LossWeights(diffeo=1.0), name="diffeo-warmup"),
                TrainingPhase(epochs=400, loss_weights=LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0), name="full"),
            ]
        """
        total_epochs = 0
        for phase in schedule:
            phase_name = phase.name or f"phase-{schedule.index(phase)}"
            print(f"\n[{model_name}] Starting {phase_name}: {phase.epochs} epochs")

            for epoch in range(phase.epochs):
                losses = self.train_epoch(data_loader, {model_name: phase.loss_weights})
                total_epochs += 1

                if (epoch + 1) % print_interval == 0:
                    print(f"  Epoch {total_epochs} ({phase_name} {epoch+1}/{phase.epochs}): "
                          f"loss = {losses[model_name]:.4f}")

        return total_epochs
     
def train_models(data_loader, trainer: MultiModelTrainer, config: TrainingConfig):
    for epoch in range(config.epochs):
        epoch_losses = trainer.train_epoch(data_loader)
        # Print progress
        if (epoch + 1) % config.print_interval == 0:
            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in epoch_losses.items()])
            print(f"Epoch {epoch + 1}, Losses - {loss_str}")
    return None
