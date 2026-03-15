#!/bin/bash
#SBATCH --job-name=mb_demo
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_demo_%j.log
#SBATCH --error=mb_demo_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.muller_brown_demo --n-seeds 5 --D 11
