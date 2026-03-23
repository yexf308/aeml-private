#!/bin/bash
#SBATCH --job-name=mfpt_C
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=mfpt_C_%j.log
#SBATCH --error=mfpt_C_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mfpt_full_ablation --n-seeds 10 --D 11 --output mfpt_ablation_C.csv
