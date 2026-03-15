#!/bin/bash
#SBATCH --job-name=mb_sweep
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_sweep_%j.log
#SBATCH --error=mb_sweep_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_contractive_sweep --n-seeds 3 --D 11
