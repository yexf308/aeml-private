#!/bin/bash
#SBATCH --job-name=nd_lam01
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=nd_sweep_lam01_%j.log
#SBATCH --error=nd_sweep_lam01_%j.log
#SBATCH --dependency=afterok:59238

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== N×D Sweep: D=201 only, lambda_smooth=0.1 ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.highd_N_D_sweep \
    --n-seeds 10 \
    --epochs 500 \
    --sde-epochs 300 \
    --lambda-smooth 0.1 \
    --d-values 201 \
    --output highd_N_D_sweep_lam01.csv

echo ""
echo "End: $(date)"
