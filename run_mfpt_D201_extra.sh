#!/bin/bash
#SBATCH --job-name=mfpt_D201x
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=mfpt_D201x_%j.log
#SBATCH --error=mfpt_D201x_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== MFPT D=201: quartic_dome + sinusoidal (no smoothness) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.mfpt_full_ablation \
    --n-seeds 10 \
    --D 201 \
    --N 50 \
    --surfaces quartic_dome sinusoidal \
    --output mfpt_ablation_noS_D201_extra.csv

echo ""
echo "End: $(date)"
echo "Done."
