#!/bin/bash
#SBATCH --job-name=paired_test
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --output=paired_test_%j.log
#SBATCH --error=paired_test_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_mfpt_paired_test
