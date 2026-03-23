#!/bin/bash
#SBATCH --job-name=abl_dn
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=ablation_deltanet_%j.log
#SBATCH --error=ablation_deltanet_%j.log

cd /network/rit/home/xy611816/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

echo "=== Table 1: Rotation ablation D=11 ==="
python3 -u -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics rotation \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_rotation_d11_dn.csv

echo ""
echo "=== Table 1: Rotation ablation D=201 ==="
python3 -u -m experiments.full_ablation --n-seeds 10 --D 201 --dynamics rotation \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_rotation_d201_dn.csv

echo ""
echo "=== Table 1: MB ablation D=11 ==="
python3 -u -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics mb \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_mb_d11_dn.csv

echo ""
echo "=== Table 1: MB ablation D=201 ==="
python3 -u -m experiments.full_ablation --n-seeds 10 --D 201 --dynamics mb \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_mb_d201_dn.csv
