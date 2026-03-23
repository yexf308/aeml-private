#!/bin/bash
#SBATCH --job-name=mfpt_3s_11
#SBATCH --partition=batch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=mb_mfpt_3surf_d11_%j.log
#SBATCH --error=mb_mfpt_3surf_d11_%j.log

cd /network/rit/lab/Yelab/aeml
source /network/rit/lab/Yelab/anaconda3/bin/activate

python3 -u -m experiments.mb_oracle_drift --n-seeds 10 --D 11 --epochs 4000 \
    --surfaces hyperbolic_paraboloid quartic_dome sinusoidal \
    --grid-res 50 --n-traj 2000 \
    --output mb_mfpt_3surf_d11.csv
