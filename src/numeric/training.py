from .autoencoders import AutoEncoder
from .losses import autoencoder_loss, LossWeights
from .datasets import DatasetBatch
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# TODO: TrainingConfig holds a latent and hidden dim size. Why should it? ModelConfig handles that.
# We aren't necessarily testing the same architectures. So it's not really a training hyperparameter.
#
# If you look below you will notice 'add_model' pulls from it for default architecture if not specified in
# the current ModelConfig instance.
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


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    loss_weights: LossWeights
    architecture_params: Dict[str, Any] | None= None

class MultiModelTrainer:
    """Class to handle training multiple models simultaneously"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models: Dict[str, AutoEncoder] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        
        self.models[model_config.name] = model
        self.optimizers[model_config.name] = optimizer
        self.model_configs[model_config.name] = model_config
    
    def create_data_loader(self, dataset: DatasetBatch | Tuple[torch.Tensor, ...]):
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
            # Assuming batch contains (samples, local_samples, mu, cov, p, weights)
            x, _, mu, cov, p, _, _ = batch
            targets = (x, mu, cov, p)
            
            for model_name, model in self.models.items():
                optimizer = self.optimizers[model_name]
                loss_weights = self.model_configs[model_name].loss_weights
                
                optimizer.zero_grad()
                loss = autoencoder_loss(model, targets, loss_weights)
                loss.backward()
                optimizer.step()
                
                epoch_losses[model_name] += loss.item()
        
        # Average losses over batches
        num_batches = len(data_loader)
        for model_name in epoch_losses:
            epoch_losses[model_name] /= num_batches
        
        return epoch_losses
     
def train_models(data_loader, trainer: MultiModelTrainer, config: TrainingConfig):
    for epoch in range(config.epochs):
        epoch_losses = trainer.train_epoch(data_loader)
        # Print progress
        if (epoch + 1) % config.print_interval == 0:
            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in epoch_losses.items()])
            print(f"Epoch {epoch + 1}, Losses - {loss_str}")
    return None
