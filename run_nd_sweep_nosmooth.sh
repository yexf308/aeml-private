#!/bin/bash
#SBATCH --job-name=nd_noS
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=nd_noS_%j.log
#SBATCH --error=nd_noS_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== N×D sweep: NO smoothness (lambda_S=0) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.highd_N_D_sweep \
    --n-seeds 10 \
    --output highd_N_D_sweep_noS.csv

echo ""
echo "End: $(date)"
