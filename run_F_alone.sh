#!/bin/bash
#SBATCH --job-name=F_alone
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=F_alone_%j.log
#SBATCH --error=F_alone_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.investigate_F_alone --n-seeds 5
