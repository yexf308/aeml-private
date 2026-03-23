#!/bin/bash
#SBATCH --job-name=mfpt_k
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=mfpt_k_ablation_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mfpt_k_ablation \
    --n-seeds 10 \
    --epochs 500 \
    --sde-epochs 300 \
    --D 11 \
    --N 50
