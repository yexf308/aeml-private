#!/bin/bash
#SBATCH --job-name=dn100_201
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=mb_mfpt_dn_n100_d201_%j.log
#SBATCH --error=mb_mfpt_dn_n100_d201_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt --n-seeds 10 --D 201 --N 100 --epochs 4000 \
    --surfaces paraboloid hyperbolic_paraboloid quartic_dome sinusoidal \
    --conditions baseline T T+F \
    --sampling delta_net \
    --n-traj 2000 \
    --output mb_mfpt_dn_n100_d201.csv
