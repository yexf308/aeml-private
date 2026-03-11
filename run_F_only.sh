#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=F_only_%j.log
#SBATCH --job-name=F_only

cd /network/rit/lab/Yelab/aeml
python3 -u -m experiments.run_F_only --n-seeds 10
