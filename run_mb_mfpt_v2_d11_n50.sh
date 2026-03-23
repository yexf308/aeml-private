#!/bin/bash
#SBATCH --job-name=mb_mfpt_n50_d11
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_mfpt_v2_d11_n50_%j.log
#SBATCH --error=mb_mfpt_v2_d11_n50_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt --n-seeds 10 --D 11 --N 50 --epochs 4000 \
    --conditions baseline T T+F --output mb_mfpt_v2_d11_n50.csv
