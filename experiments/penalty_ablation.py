import argparse

from experiments.runner import ExperimentConfig, ModelSpec, run_experiment
from src.numeric.losses import LossWeights

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description='Train autoencoders on manifold data')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=2000, help='Number of samples to generate') # TODO This one threw an error. 
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--surface', type=str, default='paraboloid', help='Surface choice')
    parser.add_argument('--rbm', action='store_true', help='Use Riemannian Brownian motion dynamics')
    parser.add_argument('--embed', action='store_true', help='Embed data into higher dimension')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Embedding dimension when --embed is set')
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        epochs=args.epochs,
        n_samples=args.n_samples,
        surface_choice=args.surface,
        rbm=args.rbm,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        test_size=0.97,
        embed=args.embed,
        embedding_dim=args.embedding_dim,
    )
    
    # Add models with different configurations
    model_configs = [
        ModelSpec(name="0th", loss_weights=LossWeights()),
        ModelSpec(name="1st-0.001", loss_weights=LossWeights(tangent_bundle=0.001)),
        ModelSpec(name="1st-0.01", loss_weights=LossWeights(tangent_bundle=0.01)),
        ModelSpec(name="1st-0.05", loss_weights=LossWeights(tangent_bundle=0.05)),
    ]

    run_experiment(config, model_configs)
    
    return None

if __name__ == "__main__":
    main()
