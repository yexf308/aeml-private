#!/bin/bash
#SBATCH --job-name=mfpt_dn
#SBATCH --partition=batch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=mb_mfpt_deltanet_d11_%j.log
#SBATCH --error=mb_mfpt_deltanet_d11_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt --n-seeds 10 --D 11 --N 100 --epochs 4000 \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --conditions baseline T T+F \
    --sampling delta_net \
    --n-traj 2000 \
    --output mb_mfpt_deltanet_d11.csv
