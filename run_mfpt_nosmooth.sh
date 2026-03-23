#!/bin/bash
#SBATCH --job-name=mfpt_noS
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=mfpt_noS_%j.log
#SBATCH --error=mfpt_noS_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== MFPT ablation: 4 surfaces, NO smoothness (lambda_S=0) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.mfpt_full_ablation \
    --n-seeds 10 \
    --D 11 \
    --N 50 \
    --output mfpt_ablation_noS.csv

echo ""
echo "=== D=201 (paraboloid + hyp_parab only) ==="
echo ""

python3 -u -m experiments.mfpt_full_ablation \
    --n-seeds 10 \
    --D 201 \
    --N 50 \
    --surfaces paraboloid hyperbolic_paraboloid \
    --output mfpt_ablation_noS_D201.csv

echo ""
echo "End: $(date)"
