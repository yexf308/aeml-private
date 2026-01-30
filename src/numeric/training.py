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
class ModelConfig:
    """Configuration for individual models"""
    name: str
    loss_weights: LossWeights
    architecture_params: Optional[Dict[str, Any]] = None

class MultiModelTrainer:
    """Class to handle training multiple models simultaneously"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.models: Dict[str, AutoEncoder] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
    
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
        """Create a DataLoader for batched training"""
        if isinstance(dataset, DatasetBatch):
            tensors = dataset.as_tuple()
        else:
            tensors = dataset
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
    
    def train_epoch(self, data_loader):
        """Train all models for one epoch"""
        epoch_losses = {}

        for model_name, model in self.models.items():
            model.train()
            epoch_losses[model_name] = 0.0

        for batch_idx, batch in enumerate(data_loader):
            # Assuming batch contains (samples, local_samples, mu, cov, p, weights, hessians)
            x, _, mu, cov, p, _, _ = batch
            # Move tensors to device
            x = x.to(self.device)
            mu = mu.to(self.device)
            cov = cov.to(self.device)
            p = p.to(self.device)
            targets = (x, mu, cov, p)

            for model_name, model in self.models.items():
                optimizer = self.optimizers[model_name]
                loss_weights = self.model_configs[model_name].loss_weights

                optimizer.zero_grad()
                loss = autoencoder_loss(model, targets, loss_weights)
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
     
def train_models(data_loader, trainer: MultiModelTrainer, config: TrainingConfig):
    for epoch in range(config.epochs):
        epoch_losses = trainer.train_epoch(data_loader)
        # Print progress
        if (epoch + 1) % config.print_interval == 0:
            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in epoch_losses.items()])
            print(f"Epoch {epoch + 1}, Losses - {loss_str}")
    return None
