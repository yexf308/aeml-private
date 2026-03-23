#!/bin/bash
#SBATCH --job-name=abl_C
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=ablation_C_%j.log
#SBATCH --error=ablation_C_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.paper_experiments ablation --n-seeds 10 --output paper_ablation_C.csv
