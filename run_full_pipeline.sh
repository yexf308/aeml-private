#!/bin/bash
#SBATCH --job-name=full_pipe
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=full_pipeline_%j.log
#SBATCH --error=full_pipeline_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== Full Pipeline Evaluation: D=3, 4 surfaces, 4 conditions ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

python3 -u -m experiments.full_pipeline_evaluation \
    --n-seeds 10 \
    --epochs 500 \
    --sde-epochs 300 \
    --lambda-smooth 0.5 \
    --output paper_full_pipeline.csv

echo ""
echo "End: $(date)"
