from .losses import l2_loss, tangent_bundle_loss, contraction_loss, diffeomorphism_penalty, fro_distance_sq
from .autoencoders import AutoEncoder
from .training import MultiModelTrainer
from .datasets import DatasetBatch
from .geometry import transform_covariance, ambient_quadratic_variation_drift, regularized_metric_inverse
import pandas as pd
import numpy as np
import torch
from scipy import stats

from itertools import combinations
from typing import Tuple, cast



def perform_pairwise_ttests(all_sample_losses):
    """Perform pairwise Welch's t-tests for each loss type"""
    model_names = list(all_sample_losses.keys())
    loss_types = all_sample_losses[model_names[0]].columns
    
    test_results = {}
    
    for loss_type in loss_types:
        test_results[loss_type] = {}
        
        # Pairwise comparisons
        for model1, model2 in combinations(model_names, 2):
            losses1 = all_sample_losses[model1][loss_type]
            losses2 = all_sample_losses[model2][loss_type]
            
            # Welch's t-test (unequal variances)
            # Null Hypothesis is both means are equal
            # Alternative is mean(loss1) > mean(loss2)
            ttest_result = stats.ttest_ind(losses1, losses2, equal_var=False, alternative="greater")
            # Pylance and TtestResult aren't talking to each other, so we have to cast.
            statistic, p_value = cast(Tuple[float, float], ttest_result)
            

            test_results[loss_type][f"{model1}_vs_{model2}"] = {
                'statistic': statistic,
                'p_value': p_value,
                'mean_diff': losses1.mean() - losses2.mean(),
                'significant_05': p_value < 0.05,
                'significant_01': p_value < 0.01,
                'model1_mean': losses1.mean(),
                'model2_mean': losses2.mean(),
                'model1_std': losses1.std(),
                'model2_std': losses2.std()
            }
    
    return test_results

# TODO: Make a version of this for producing LaTeX tables.
def print_ttest_results(summary_stats, statistical_tests, alpha=0.05):
    """Print formatted results
    
    Arguments:
        summary_stats: the mean, std, sem of each loss
        statistical_tests: output from 'perform_pairwise_ttests'
        alpha: sig-lvl
    """
    print("="*80)
    print("MODEL COMPARISON WITH WELCH'S T-TESTS")
    print("="*80)
    
    # Summary statistics
    print("\nMODEL SUMMARY (Mean ± Std):")
    print("-" * 40)
    for model_name, stats in summary_stats.items():
        print(f"\n{model_name}:")
        for loss_type in stats['mean'].index:
            mean = stats['mean'][loss_type]
            std = stats['std'][loss_type]
            sem = stats['sem'][loss_type]
            print(f"  {loss_type:15s}: {mean:.6f} ± {std:.6f} (SEM: {sem:.6f})")
    
    # T-test results
    print(f"\nPAIRWISE T-TESTS (α = {alpha}):")
    print("NULL HYPOTHESIS: E(Loss 1) = E(Loss 2)")
    print("ALTERNATIVE HYPOTHESIS: E(Loss 1) > E(Loss 2)")
    print("-" * 60)
    
    for loss_type, comparisons in statistical_tests.items():
        print(f"\n{loss_type.upper()} LOSS:")
        print(f"{'Comparison':<50} {'t-stat':<8} {'p-value':<10} {'Sig':<5} {'Mean_diff':<12}")
        print("-" * 75)
        
        for comparison, result in comparisons.items():
            significance = "***" if result['p_value'] < 0.001 else \
                         "**" if result['p_value'] < 0.01 else \
                         "*" if result['p_value'] < 0.05 else "ns"
            
            print(f"{comparison:<50} {result['statistic']:7.3f} {result['p_value']:9.4f} "
                  f"{significance:<5} {result['mean_diff']:11.6f}")
            

def compute_losses_per_sample(model: AutoEncoder, targets: DatasetBatch):
    """
        Compute losses for each individual sample using your existing loss functions
    """
    x = targets.samples
    mu = targets.mu
    cov = targets.cov
    p = targets.p
    hessians = targets.hessians
    
    z = model.encoder(x)
    x_hat = model.decoder(z)
    phat = model.orthogonal_projection(z)
    dpi = model.jacobian_encoder(x)
    dphi = model.jacobian_decoder(z)
    d2phi = model.hessian_decoder(z)
    g_perf = dphi.mT @ dphi
    ginv_perf = regularized_metric_inverse(g_perf)
    dphi_penroseinv = ginv_perf @ dphi.mT
    local_cov = transform_covariance(cov, dphi_penroseinv)
    qhat = ambient_quadratic_variation_drift(local_cov, d2phi)
    nhat = torch.eye(p.size(1), device=p.device, dtype=p.dtype).unsqueeze(0) - phat
    ntrue = torch.eye(p.size(1), device=p.device, dtype=p.dtype).unsqueeze(0) - p
    normal_drift_true = torch.bmm(ntrue, mu.unsqueeze(-1)).squeeze(-1)
    normal_drift_model = torch.bmm(nhat, 0.5 * qhat.unsqueeze(-1)).squeeze(-1)
    # TODO compute tangent drift error

    
    # Use your individual loss functions (before empirical averaging)
    recon_losses = l2_loss(x_hat, x)  # Shape: (batch_size,)
    tangent_losses = tangent_bundle_loss(phat, p)  # Shape: (batch_size,) 
    contraction_losses = contraction_loss(dpi)  # Shape: (batch_size,)
    contraction_losses_decoder = contraction_loss(dphi)  # Shape: (batch_size,)
    diffeo_losses = diffeomorphism_penalty(dpi, dphi) # Shape: (batch_size,)
    curvature_losses = l2_loss(normal_drift_model, normal_drift_true)  # Shape: (batch_size,)
    
    # Compute hessian loss for each hessian, averaged over the samples
    hessian_losses = torch.max(fro_distance_sq(d2phi, hessians), dim=-1).values
    # Create DataFrame with one row per sample
    losses_df = pd.DataFrame({
        'reconstruction': recon_losses.detach().cpu().numpy(),
        'tangent': tangent_losses.detach().cpu().numpy(), 
        'encoder contr': contraction_losses.detach().cpu().numpy(),
        'decoder contr': contraction_losses_decoder.detach().cpu().numpy(),
        "diffeo": diffeo_losses.detach().cpu().numpy(),
        "curvature": curvature_losses.detach().cpu().numpy(),
        "hessian": hessian_losses.detach().cpu().numpy()
    })

    return losses_df

def evaluate_models_with_ttests(trainer: MultiModelTrainer, test_data: DatasetBatch):
    """Evaluate all models and perform pairwise t-tests to see whether one mean loss is greater or not.
    """
    targets = test_data
    
    # Get individual sample losses for each model
    all_sample_losses = {}
    
    for model_name, model in trainer.models.items():
        model.eval()
        sample_losses = compute_losses_per_sample(model, targets=targets)
        all_sample_losses[model_name] = sample_losses
    
    # Perform pairwise t-tests
    statistical_tests = perform_pairwise_ttests(all_sample_losses)
    
    # Also compute summary statistics (means across samples)
    summary_stats = {}
    for model_name, losses in all_sample_losses.items():
        summary_stats[model_name] = {
            'mean': losses.mean(),
            'std': losses.std(),
            'sem': losses.sem()  # Standard error of mean
        }
    
    return summary_stats, statistical_tests, all_sample_losses
