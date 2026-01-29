from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp
import torch
import torch.nn as nn

from src.numeric.datagen import (
    create_embedding_matrix,
    embed_dataset_with_qr_matrix,
    sample_from_manifold,
    train_test_split_dataset,
)
from src.numeric.datasets import DatasetBatch
from src.numeric.losses import LossWeights
from src.numeric.performance_stats import evaluate_models_with_ttests, print_ttest_results
from src.numeric.training import ModelConfig, MultiModelTrainer, TrainingConfig, train_models
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import (
    gaussian_bump,
    hyperbolic_paraboloid,
    monkey_saddle,
    one_sheet_hyperboloid,
    paraboloid,
    plane,
    sinusoidal,
    surface,
)


@dataclass
class ExperimentConfig:
    epochs: int = 2000
    n_samples: int = 2000
    input_dim: int = 3
    hidden_dim: int = 64
    latent_dim: int = 2
    learning_rate: float = 0.005
    batch_size: int = 20
    test_size: float = 0.97
    print_interval: int = 100
    embed: bool = False
    embedding_dim: int = 50
    embedding_seed: int = 17
    data_seed: int = 42
    surface_choice: str = "paraboloid"
    rbm: bool = False
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0))


@dataclass
class ModelSpec:
    name: str
    loss_weights: LossWeights
    architecture_params: Optional[Dict[str, object]] = None


_SURFACE_MAP: Dict[str, Callable[[sp.Symbol, sp.Symbol], sp.Expr]] = {
    "paraboloid": paraboloid,
    "hyperbolic paraboloid": hyperbolic_paraboloid,
    "hyperboloid": one_sheet_hyperboloid,
    "monkey saddle": monkey_saddle,
    "gaussian bump": gaussian_bump,
    "sinusoidal": sinusoidal,
    "plane": plane,
}


def _make_arch_params(config: ExperimentConfig, hidden_dims: Optional[Sequence[int]] = None) -> Dict[str, object]:
    dims = list(hidden_dims) if hidden_dims is not None else [config.hidden_dim]
    return {
        "extrinsic_dim": config.embedding_dim if config.embed else config.input_dim,
        "intrinsic_dim": config.latent_dim,
        "hidden_dims": dims,
        "encoder_act": nn.Tanh(),
        "decoder_act": nn.Tanh(),
    }


def setup_manifold_and_data(config: ExperimentConfig) -> Tuple[DatasetBatch, DatasetBatch]:
    u, v = sp.symbols("u v", real=True)

    if config.surface_choice not in _SURFACE_MAP:
        choices = ", ".join(sorted(_SURFACE_MAP))
        raise ValueError(
            f"Invalid surface name. Available surfaces are: {choices}"
        )

    local_coord, chart = surface(_SURFACE_MAP[config.surface_choice], u, v)

    if config.rbm:
        local_drift = None
        local_diffusion = None
        print("Generating a Riemannian Brownian motion")
    else:
        print("Generating a non-isotropic manifold SDE")
        local_drift = sp.Matrix([-v, u])
        local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])

    manifold = RiemannianManifold(local_coord, chart)
    manifold_sde = ManifoldSDE(
        manifold,
        local_drift=local_drift,
        local_diffusion=local_diffusion,
    )

    print("Manifold properties:")
    print("Chart", manifold_sde.manifold.chart)
    print("Induced metric tensor", manifold_sde.manifold.metric_tensor())
    print("Manifold SDE Properties:")
    print("Local drift:", manifold_sde.local_drift)
    print("Local diffusion:", manifold_sde.local_diffusion)
    print("Local covariance:", manifold_sde.local_covariance)
    print("Ambient drift:", manifold_sde.ambient_drift)
    print("Ambient diffusion:", manifold_sde.ambient_diffusion)
    print("Ambient covariance:", manifold_sde.ambient_covariance)

    dataset = sample_from_manifold(
        manifold_sde,
        list(config.bounds),
        n_samples=config.n_samples,
        seed=config.data_seed,
    )

    if config.embed:
        print("Embedding into higher dimension isometrically and randomly")
        print("Drift shape before embedding")
        print(dataset.mu.size())
        embedding_matrix = create_embedding_matrix(
            config.embedding_dim, config.input_dim, config.embedding_seed
        )
        dataset = embed_dataset_with_qr_matrix(dataset, embedding_matrix)
        print("Drift shape after embedding")
        print(dataset.mu.size())

    print("\nData shapes:")
    print(f"Samples: {dataset.samples.shape}")
    print(f"Local samples: {dataset.local_samples.shape}")
    print(f"Weights: {dataset.weights.shape}")
    print(f"Mu: {dataset.mu.shape}")
    print(f"Cov: {dataset.cov.shape}")
    print(f"P: {dataset.p.shape}")
    print(f"Hessians: {dataset.hessians.shape}")
    print(f"Samples type: {type(dataset.samples)}")
    print(f"Samples size: {dataset.samples.size()}")

    train_data, test_data = train_test_split_dataset(dataset, test_size=config.test_size)

    print("\nSizes after train-test-split:")
    print(f"Train samples: {train_data.samples.size()}")
    print(f"Test samples: {test_data.samples.size()}")

    return train_data, test_data


def run_experiment(config: ExperimentConfig, model_specs: List[ModelSpec]) -> None:
    train_data, test_data = setup_manifold_and_data(config)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=config.epochs,
        n_samples=config.n_samples,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        test_size=config.test_size,
        print_interval=config.print_interval,
        embed=config.embed,
        embedding_dim=config.embedding_dim,
        embedding_seed=config.embedding_seed,
        effective_dim=config.embedding_dim if config.embed else config.input_dim,
    ))

    for spec in model_specs:
        arch_params = spec.architecture_params or _make_arch_params(config)
        trainer.add_model(ModelConfig(
            name=spec.name,
            loss_weights=spec.loss_weights,
            architecture_params=arch_params,
        ))

    data_loader = trainer.create_data_loader(train_data)

    print(f"\nTraining {len(trainer.models)} models...")
    train_models(data_loader, trainer, trainer.config)

    print("\nEvaluating models on test set...")
    summary_stats, statistical_tests, _ = evaluate_models_with_ttests(trainer, test_data)
    print_ttest_results(summary_stats, statistical_tests)
