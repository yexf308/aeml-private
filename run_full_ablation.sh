#!/bin/bash
# Submit all 4 ablation jobs

# D=11 rotation (~8h)
sbatch --job-name=abl_rot11 --partition=dgx --gres=gpu:1 --time=12:00:00 --mem=32G \
    --output=full_ablation_rot_d11_%j.log --error=full_ablation_rot_d11_%j.log \
    --wrap="cd /network/rit/lab/Yelab/aeml && source /network/rit/lab/Yelab/anaconda3/bin/activate && python3 -u -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics rotation --output full_ablation_rotation_d11.csv"

# D=11 MB (~24h)
sbatch --job-name=abl_mb11 --partition=dgx --gres=gpu:1 --time=36:00:00 --mem=32G \
    --output=full_ablation_mb_d11_%j.log --error=full_ablation_mb_d11_%j.log \
    --wrap="cd /network/rit/lab/Yelab/aeml && source /network/rit/lab/Yelab/anaconda3/bin/activate && python3 -u -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics mb --output full_ablation_mb_d11.csv"

# D=201 rotation (~12h)
sbatch --job-name=abl_rot201 --partition=dgx --gres=gpu:1 --time=18:00:00 --mem=32G \
    --output=full_ablation_rot_d201_%j.log --error=full_ablation_rot_d201_%j.log \
    --wrap="cd /network/rit/lab/Yelab/aeml && source /network/rit/lab/Yelab/anaconda3/bin/activate && python3 -u -m experiments.full_ablation --n-seeds 10 --D 201 --dynamics rotation --output full_ablation_rotation_d201.csv"

# D=201 MB (~48h)
sbatch --job-name=abl_mb201 --partition=dgx --gres=gpu:1 --time=72:00:00 --mem=32G \
    --output=full_ablation_mb_d201_%j.log --error=full_ablation_mb_d201_%j.log \
    --wrap="cd /network/rit/lab/Yelab/aeml && source /network/rit/lab/Yelab/anaconda3/bin/activate && python3 -u -m experiments.full_ablation --n-seeds 10 --D 201 --dynamics mb --output full_ablation_mb_d201.csv"
