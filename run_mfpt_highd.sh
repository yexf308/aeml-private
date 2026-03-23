#!/bin/bash
#SBATCH --job-name=mfpt_d201
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=mfpt_highd_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mfpt_full_ablation \
    --n-seeds 10 \
    --epochs 500 \
    --sde-epochs 300 \
    --D 201 \
    --N 50 \
    --output mfpt_ablation_D201.csv
