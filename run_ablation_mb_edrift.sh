#!/bin/bash
#SBATCH --job-name=abl_mb_ed
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=ablation_mb_edrift_%j.log
#SBATCH --error=ablation_mb_edrift_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

echo "=== MB ablation D=11 (with E_drift) ==="
python3 -u -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics mb \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_mb_d11_dn.csv

echo ""
echo "=== MB ablation D=201 (with E_drift) ==="
python3 -u -m experiments.full_ablation --n-seeds 10 --D 201 --dynamics mb \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_mb_d201_dn.csv
