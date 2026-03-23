#!/bin/bash
#SBATCH --job-name=cf_rot_11
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=coeff_rotation_d11_%j.log
#SBATCH --error=coeff_rotation_d11_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics rotation \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --output full_ablation_rotation_d11_dn.csv
