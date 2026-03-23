#!/bin/bash
#SBATCH --job-name=mfpt_n50
#SBATCH --partition=batch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=mb_mfpt_n50_d11_%j.log
#SBATCH --error=mb_mfpt_n50_d11_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt --n-seeds 10 --D 11 --N 50 --epochs 4000 \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --conditions baseline T T+F \
    --n-traj 2000 --uniform \
    --output mb_mfpt_n50_d11.csv
