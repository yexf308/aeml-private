#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=drift_capacity_%j.log
#SBATCH --job-name=drift_cap

cd /network/rit/lab/Yelab/aeml
python3 -u -m experiments.test_drift_capacity --n-seeds 5
