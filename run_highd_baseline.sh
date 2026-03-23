#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=highd_baseline_%j.log
#SBATCH --job-name=highd_bl

cd /network/rit/lab/Yelab/aeml
python3 -u -m experiments.highd_baseline_comparison --n-seeds 10
