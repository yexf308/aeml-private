#!/bin/bash
#SBATCH --job-name=nd_sweep
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=nd_sweep_%j.log
#SBATCH --error=nd_sweep_%j.log

cd /network/rit/lab/Yelab/aeml

# Activate conda if needed
source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== N×D Sweep: sqrt(D) K weight + Jacobian smoothing (D-normalised) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(which python3)"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.highd_N_D_sweep \
    --n-seeds 10 \
    --epochs 500 \
    --sde-epochs 300 \
    --lambda-smooth 0.5 \
    --output highd_N_D_sweep.csv

echo ""
echo "End: $(date)"
