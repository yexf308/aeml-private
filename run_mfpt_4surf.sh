#!/bin/bash
#SBATCH --job-name=mfpt_4surf
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=mfpt_4surf_%j.log
#SBATCH --error=mfpt_4surf_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== MFPT ablation: 4 surfaces (paraboloid, hyp_parab, quartic_dome, sinusoidal) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.mfpt_full_ablation \
    --n-seeds 10 \
    --D 11 \
    --N 50 \
    --output mfpt_ablation_4surf.csv

echo ""
echo "End: $(date)"
