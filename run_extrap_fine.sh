#!/bin/bash
#SBATCH --job-name=extrap_fine
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=extrap_fine_%j.log
#SBATCH --error=extrap_fine_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== Extrapolation (fine resolution, delta<=0.3) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.paper_experiments extrapolation \
    --n-seeds 10 \
    --output paper_extrapolation.csv

echo ""
echo "=== Dynamics extrapolation (fine resolution, delta<=0.3) ==="
echo ""

python3 -u -m experiments.paper_experiments dynamics \
    --n-seeds 10 \
    --output paper_dynamics.csv

echo ""
echo "End: $(date)"
