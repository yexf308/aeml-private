#!/bin/bash
#SBATCH --job-name=mb_nd_sweep
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_nd_sweep_%j.log
#SBATCH --error=mb_nd_sweep_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_nd_sweep --n-seeds 10
