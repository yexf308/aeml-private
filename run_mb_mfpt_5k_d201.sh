#!/bin/bash
#SBATCH --job-name=mb_mfpt_5k_d201
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_mfpt_5k_d201_%j.log
#SBATCH --error=mb_mfpt_5k_d201_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt --n-seeds 10 --D 201 --n-traj 5000 \
    --conditions baseline T T+F --output mb_mfpt_5k_d201.csv
