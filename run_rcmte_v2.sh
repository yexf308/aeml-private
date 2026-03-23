#!/bin/bash
#SBATCH --job-name=rcmte_v2
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=rcmte_v2_%j.log
#SBATCH --error=rcmte_v2_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== rcMTE v2 (stopped mask) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

# D=11, 10 seeds
python3 -u -m experiments.rcmte_comparison \
    --n-seeds 10 --D 11 --N 20 \
    --output rcmte_v2_D11.csv

echo ""
echo "=== D=201 ==="
echo ""

# D=201, 10 seeds
python3 -u -m experiments.rcmte_comparison \
    --n-seeds 10 --D 201 --N 20 \
    --output rcmte_v2_D201.csv

echo ""
echo "End: $(date)"
