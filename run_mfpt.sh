#!/bin/bash
#SBATCH --job-name=mfpt_cmp
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=mfpt_comparison_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mfpt_comparison \
    --n-seeds 5 \
    --epochs 500 \
    --sde-epochs 300 \
    --D 11 \
    --N 50 \
    --surface paraboloid
