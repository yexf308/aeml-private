#!/bin/bash
#SBATCH --job-name=smooth_abl
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=smooth_abl_%j.log
#SBATCH --error=smooth_abl_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== Drift smoothness ablation: T+F with vs without smoothness ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.drift_smoothness_ablation \
    --n-seeds 10 \
    --D 11 \
    --N 50 \
    --output drift_smooth_ablation.csv

echo ""
echo "End: $(date)"
