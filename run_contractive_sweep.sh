#!/bin/bash
#SBATCH --job-name=c_sweep
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=contractive_sweep_%j.log
#SBATCH --error=contractive_sweep_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.contractive_sweep
