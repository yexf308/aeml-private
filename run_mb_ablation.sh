#!/bin/bash
#SBATCH --job-name=mb_ablation
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=mb_ablation_%j.log
#SBATCH --error=mb_ablation_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_ablation --n-seeds 10 --D 11
