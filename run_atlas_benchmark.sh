#!/bin/bash
#SBATCH --job-name=atlas_bm
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=atlas_benchmark_%j.log
#SBATCH --error=atlas_benchmark_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.atlas_benchmark --n-seeds 10 --D 11 --N 200 \
    --mode oracle --surfaces paraboloid \
    --output atlas_benchmark_d11.csv
