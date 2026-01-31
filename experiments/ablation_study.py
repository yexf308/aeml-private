"""
Ablation Study for AEML Loss Penalties

This script systematically tests different combinations of loss penalties
across multiple benchmark surfaces to understand their individual and
combined effects on model performance.

Penalties tested (6 total, 64 combinations):
- T: tangent_bundle - Tangent space alignment
- C: contractive - Encoder Jacobian regularization
- N: normal_decoder_jacobian - Normal component of decoder Jacobian
- D: decoder_contraction - Decoder Jacobian regularization
- F: diffeo - Diffeomorphism penalty (encoder-decoder composition)
- K: curvature - Normal drift matching (second-order)

Surfaces tested:
- paraboloid, hyperbolic paraboloid, hyperboloid
- monkey saddle, gaussian bump, sinusoidal, plane

Usage:
    # Run curated subset of penalties on specific surfaces
    python -m experiments.ablation_study --surfaces paraboloid "gaussian bump" --epochs 500

    # Run ALL 64 penalty combinations on specific surfaces
    python -m experiments.ablation_study --surfaces paraboloid --all-penalties --epochs 500

    # Run on all surfaces with all penalties (7 × 64 = 448 experiments)
    python -m experiments.ablation_study --full --all-penalties --epochs 500

    # Run specific penalties
    python -m experiments.ablation_study --penalties T T+K T+F+K baseline
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from experiments.runner import ExperimentConfig, ModelSpec, _SURFACE_MAP, setup_manifold_and_data
from src.numeric.losses import LossWeights
from src.numeric.performance_stats import compute_losses_per_sample, evaluate_models_with_ttests
from src.numeric.training import ModelConfig, MultiModelTrainer, TrainingConfig, train_models


# Penalty short names and their default weights
PENALTY_WEIGHTS = {
    "T": ("tangent_bundle", 1.0),      # Tangent space alignment
    "C": ("contractive", 0.01),         # Encoder contraction
    "N": ("normal_decoder_jacobian", 0.01),  # Normal decoder jacobian
    "D": ("decoder_contraction", 0.01), # Decoder contraction
    "F": ("diffeo", 1.0),               # Diffeomorphism constraint
    "K": ("curvature", 1.0),            # Curvature drift matching
}


def generate_all_penalty_configs() -> Dict[str, LossWeights]:
    """Generate all 2^6 = 64 combinations of penalties."""
    configs = {"baseline": LossWeights()}  # Start with baseline

    penalty_keys = list(PENALTY_WEIGHTS.keys())
    n_penalties = len(penalty_keys)

    # Generate all 2^n - 1 non-empty subsets
    for mask in range(1, 2**n_penalties):
        # Build config name and weights dict
        active_penalties = []
        weights_dict = {}

        for i, key in enumerate(penalty_keys):
            if mask & (1 << i):
                active_penalties.append(key)
                param_name, weight_value = PENALTY_WEIGHTS[key]
                weights_dict[param_name] = weight_value

        config_name = "+".join(active_penalties)
        configs[config_name] = LossWeights(**weights_dict)

    return configs


# Generate all combinations
PENALTY_CONFIGS_ALL = generate_all_penalty_configs()

# Curated subset for quick tests (original selection)
PENALTY_CONFIGS_CURATED = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "C": LossWeights(contractive=0.01),
    "D": LossWeights(decoder_contraction=0.01),
    "F": LossWeights(diffeo=1.0),
    "K": LossWeights(curvature=1.0),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=1.0),
    "T+C": LossWeights(tangent_bundle=1.0, contractive=0.01),
    "F+K": LossWeights(diffeo=1.0, curvature=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
    "T+C+F+K": LossWeights(tangent_bundle=1.0, contractive=0.01, diffeo=1.0, curvature=1.0),
}

# Default to curated for backward compatibility
PENALTY_CONFIGS = PENALTY_CONFIGS_CURATED


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    surface: str
    penalty_config: str
    loss_weights: Dict[str, float]
    epochs: int
    final_losses: Dict[str, float]  # Mean losses at end of training
    test_metrics: Dict[str, Dict[str, float]]  # Per-metric mean/std
    training_time_seconds: float


def run_single_ablation(
    surface: str,
    penalty_name: str,
    loss_weights: LossWeights,
    config: ExperimentConfig,
    verbose: bool = True,
) -> AblationResult:
    """Run a single ablation experiment."""
    import time

    start_time = time.time()

    # Update config for this surface
    config.surface_choice = surface

    if verbose:
        print(f"\n{'='*60}")
        print(f"Surface: {surface} | Penalty: {penalty_name}")
        print(f"{'='*60}")

    # Setup data
    train_data, test_data = setup_manifold_and_data(config)

    # Create trainer
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
        device=config.device,
    ))

    # Add single model with specified loss weights
    trainer.add_model(ModelConfig(
        name=penalty_name,
        loss_weights=loss_weights,
    ))

    # Train
    data_loader = trainer.create_data_loader(train_data)

    final_losses = {}
    for epoch in range(config.epochs):
        epoch_losses = trainer.train_epoch(data_loader)
        if (epoch + 1) % config.print_interval == 0 and verbose:
            loss_str = ", ".join([f"{name}: {loss:.6f}" for name, loss in epoch_losses.items()])
            print(f"Epoch {epoch + 1}: {loss_str}")
        final_losses = epoch_losses

    # Evaluate on test set
    trainer.models[penalty_name].eval()
    with torch.no_grad():
        test_losses_df = compute_losses_per_sample(trainer.models[penalty_name], test_data)

    test_metrics = {
        col: {
            "mean": float(test_losses_df[col].mean()),
            "std": float(test_losses_df[col].std()),
            "median": float(test_losses_df[col].median()),
        }
        for col in test_losses_df.columns
    }

    elapsed = time.time() - start_time

    return AblationResult(
        surface=surface,
        penalty_config=penalty_name,
        loss_weights=asdict(loss_weights),
        epochs=config.epochs,
        final_losses={k: float(v) for k, v in final_losses.items()},
        test_metrics=test_metrics,
        training_time_seconds=elapsed,
    )


def run_ablation_study(
    surfaces: List[str],
    penalty_configs: Dict[str, LossWeights],
    config: ExperimentConfig,
    output_dir: str = "ablation_results",
    verbose: bool = True,
) -> List[AblationResult]:
    """Run full ablation study across surfaces and penalty configs."""

    os.makedirs(output_dir, exist_ok=True)
    results = []

    total_experiments = len(surfaces) * len(penalty_configs)
    current = 0

    for surface in surfaces:
        for penalty_name, loss_weights in penalty_configs.items():
            current += 1
            print(f"\n[{current}/{total_experiments}] Running {surface} + {penalty_name}")

            try:
                result = run_single_ablation(
                    surface=surface,
                    penalty_name=penalty_name,
                    loss_weights=loss_weights,
                    config=config,
                    verbose=verbose,
                )
                results.append(result)

                # Save intermediate results
                save_results(results, output_dir)

            except Exception as e:
                print(f"ERROR in {surface} + {penalty_name}: {e}")
                import traceback
                traceback.print_exc()

    return results


def save_results(results: List[AblationResult], output_dir: str):
    """Save results to JSON and CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results as JSON
    json_path = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Create summary DataFrame
    rows = []
    for r in results:
        row = {
            "surface": r.surface,
            "penalty": r.penalty_config,
            "epochs": r.epochs,
            "time_sec": r.training_time_seconds,
        }
        # Add test metrics
        for metric, values in r.test_metrics.items():
            row[f"{metric}_mean"] = values["mean"]
            row[f"{metric}_std"] = values["std"]
        # Add loss weights
        for k, v in r.loss_weights.items():
            row[f"weight_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"ablation_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    # Also save a "latest" version
    df.to_csv(os.path.join(output_dir, "ablation_summary_latest.csv"), index=False)

    print(f"\nResults saved to {output_dir}/")


def print_summary(results: List[AblationResult]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    # Group by surface
    surfaces = sorted(set(r.surface for r in results))
    penalties = sorted(set(r.penalty_config for r in results))

    # Print reconstruction loss comparison
    print("\nReconstruction Loss (mean):")
    print("-" * 60)
    header = f"{'Penalty':<20}" + "".join(f"{s:<12}" for s in surfaces)
    print(header)
    print("-" * 60)

    for penalty in penalties:
        row = f"{penalty:<20}"
        for surface in surfaces:
            matching = [r for r in results if r.surface == surface and r.penalty_config == penalty]
            if matching:
                val = matching[0].test_metrics.get("reconstruction", {}).get("mean", float("nan"))
                row += f"{val:<12.6f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    # Print tangent loss comparison
    print("\nTangent Bundle Loss (mean):")
    print("-" * 60)
    print(header)
    print("-" * 60)

    for penalty in penalties:
        row = f"{penalty:<20}"
        for surface in surfaces:
            matching = [r for r in results if r.surface == surface and r.penalty_config == penalty]
            if matching:
                val = matching[0].test_metrics.get("tangent", {}).get("mean", float("nan"))
                row += f"{val:<12.6f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    # Print curvature loss comparison
    print("\nCurvature Loss (mean):")
    print("-" * 60)
    print(header)
    print("-" * 60)

    for penalty in penalties:
        row = f"{penalty:<20}"
        for surface in surfaces:
            matching = [r for r in results if r.surface == surface and r.penalty_config == penalty]
            if matching:
                val = matching[0].test_metrics.get("curvature", {}).get("mean", float("nan"))
                row += f"{val:<12.6f}"
            else:
                row += f"{'N/A':<12}"
        print(row)


def run_pairwise_comparison(
    surface: str,
    config: ExperimentConfig,
    penalty_configs: Dict[str, LossWeights],
    verbose: bool = True,
):
    """Run all penalty configs on a single surface and do pairwise t-tests."""
    config.surface_choice = surface

    print(f"\n{'='*60}")
    print(f"Pairwise Comparison on: {surface}")
    print(f"{'='*60}")

    # Setup data (once for all models)
    train_data, test_data = setup_manifold_and_data(config)

    # Create trainer with all models
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
        device=config.device,
    ))

    # Add all models
    for penalty_name, loss_weights in penalty_configs.items():
        trainer.add_model(ModelConfig(
            name=penalty_name,
            loss_weights=loss_weights,
        ))

    # Train all models together
    data_loader = trainer.create_data_loader(train_data)
    print(f"\nTraining {len(trainer.models)} models...")
    train_models(data_loader, trainer, trainer.config)

    # Evaluate with t-tests
    print("\nEvaluating models on test set...")
    from src.numeric.performance_stats import print_ttest_results
    summary_stats, statistical_tests, _ = evaluate_models_with_ttests(trainer, test_data)
    print_ttest_results(summary_stats, statistical_tests)

    return summary_stats, statistical_tests


def main():
    parser = argparse.ArgumentParser(description="AEML Ablation Study")
    parser.add_argument(
        "--surfaces",
        nargs="+",
        default=["paraboloid", "gaussian bump"],
        choices=list(_SURFACE_MAP.keys()),
        help="Surfaces to test"
    )
    parser.add_argument(
        "--penalties",
        nargs="+",
        default=None,
        help="Specific penalty configs to test (e.g., 'T' 'T+K' 'baseline')"
    )
    parser.add_argument(
        "--all-penalties",
        action="store_true",
        help="Test all 64 penalty combinations (2^6)"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Run on all surfaces")
    parser.add_argument("--pairwise", action="store_true", help="Run pairwise t-test comparison")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup config
    config = ExperimentConfig(
        epochs=args.epochs,
        n_samples=args.n_samples,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        print_interval=args.epochs // 10 if args.epochs >= 10 else 1,
        data_seed=args.seed,
        device=device,
    )

    # Determine surfaces
    if args.full:
        surfaces = list(_SURFACE_MAP.keys())
    else:
        surfaces = args.surfaces

    # Determine penalties
    if args.all_penalties:
        penalty_configs = PENALTY_CONFIGS_ALL
        print(f"\n  Running FULL ablation: {len(penalty_configs)} penalty combinations")
    elif args.penalties:
        # Use specified penalties from ALL configs
        penalty_configs = {k: PENALTY_CONFIGS_ALL[k] for k in args.penalties if k in PENALTY_CONFIGS_ALL}
        if not penalty_configs:
            print(f"ERROR: None of the specified penalties found. Available: {list(PENALTY_CONFIGS_ALL.keys())[:10]}...")
            return
    else:
        penalty_configs = PENALTY_CONFIGS_CURATED

    print(f"\nAblation Study Configuration:")
    print(f"  Surfaces: {surfaces}")
    print(f"  Penalties: {list(penalty_configs.keys())}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Device: {device}")

    if args.pairwise:
        # Run pairwise comparison on each surface
        for surface in surfaces:
            run_pairwise_comparison(
                surface=surface,
                config=config,
                penalty_configs=penalty_configs,
                verbose=not args.quiet,
            )
    else:
        # Run full ablation study
        results = run_ablation_study(
            surfaces=surfaces,
            penalty_configs=penalty_configs,
            config=config,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )

        # Print summary
        print_summary(results)

        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
