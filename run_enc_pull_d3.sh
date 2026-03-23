#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=enc_pull_d3_%j.log
#SBATCH --job-name=enc_pull_d3

cd /network/rit/lab/Yelab/aeml
python3 -u -m experiments.enc_pull_d3_dynamics --n-seeds 10
