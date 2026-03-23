#!/bin/bash
#SBATCH --job-name=mb_mfpt201r
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=09:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_mfpt_oracle_d201_resume_%j.log
#SBATCH --error=mb_mfpt_oracle_d201_resume_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_oracle_drift --n-seeds 8 --D 201 --epochs 4000 \
    --surfaces paraboloid --grid-res 50 --base-seed 2042 \
    --n-traj 2000 \
    --output mb_oracle_drift_d201_resume.csv
