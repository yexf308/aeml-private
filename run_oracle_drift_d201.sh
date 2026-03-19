#!/bin/bash
#SBATCH --job-name=oracle_d201
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=oracle_drift_d201_%j.log
#SBATCH --error=oracle_drift_d201_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_oracle_drift --n-seeds 5 --D 201 --epochs 4000 \
    --surfaces paraboloid --grid-res 50 \
    --output mb_oracle_drift_d201.csv
