#!/bin/bash
#SBATCH --job-name=oracle_drift
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=oracle_drift_%j.log
#SBATCH --error=oracle_drift_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_oracle_drift --n-seeds 10 --D 11 --epochs 4000 \
    --surfaces paraboloid hyperbolic_paraboloid --grid-res 50 \
    --output mb_oracle_drift_d11.csv
