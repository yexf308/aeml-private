#!/bin/bash
#SBATCH --job-name=mfpt_rot
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=mfpt_rotation_deltanet_%j.log
#SBATCH --error=mfpt_rotation_deltanet_%j.log

cd /network/rit/home/xy611816/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

echo "=== Table 2: Rotation MFPT D=11 ==="
python3 -u -m experiments.mfpt_full_ablation --n-seeds 10 --D 11 --N 50 \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --conditions baseline T F C T+F \
    --sampling delta_net \
    --output mfpt_rotation_d11_dn.csv

echo ""
echo "=== Table 2: Rotation MFPT D=201 ==="
python3 -u -m experiments.mfpt_full_ablation --n-seeds 10 --D 201 --N 50 \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --conditions baseline T F C T+F \
    --sampling delta_net \
    --output mfpt_rotation_d201_dn.csv
