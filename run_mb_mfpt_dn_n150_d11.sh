#!/bin/bash
#SBATCH --job-name=dn150_11
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=mb_mfpt_dn_n150_d11_%j.log
#SBATCH --error=mb_mfpt_dn_n150_d11_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt --n-seeds 10 --D 11 --N 150 --epochs 4000 \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --conditions baseline T T+F \
    --sampling delta_net \
    --n-traj 2000 \
    --output mb_mfpt_dn_n150_d11.csv
