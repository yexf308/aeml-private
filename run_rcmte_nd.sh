#!/bin/bash
#SBATCH --job-name=rcmte_nd
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=rcmte_nd_%j.log
#SBATCH --error=rcmte_nd_%j.log

cd /network/rit/lab/Yelab/aeml

source ~/.bashrc
conda activate aeml 2>/dev/null || true

echo "=== rcMTE N×D sweep (enc-pull vs dec-side, no smoothness) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

OUTPUT=rcmte_nd_sweep.csv
FIRST=1

for D in 11 201; do
  for N in 20 50 100 200; do
    echo "=== D=$D, N=$N ==="
    if [ $FIRST -eq 1 ]; then
      python3 -u -m experiments.rcmte_comparison \
          --n-seeds 10 --D $D --N $N \
          --output $OUTPUT
      FIRST=0
    else
      python3 -u -m experiments.rcmte_comparison \
          --n-seeds 10 --D $D --N $N \
          --output ${OUTPUT}.tmp
      # Append without header
      tail -n +2 ${OUTPUT}.tmp >> $OUTPUT
      rm -f ${OUTPUT}.tmp
    fi
    echo ""
  done
done

echo "End: $(date)"
echo "Done."
