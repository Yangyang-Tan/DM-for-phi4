#!/bin/bash
# Sequential sampling for multiple epochs on cuda:2
# Estimated total time: 5-10 hours

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

cd /data/tyywork/DM/3Dphi4

EPOCHS=("epoch=0599" "epoch=0699" "epoch=0799")

for EP in "${EPOCHS[@]}"; do
    echo "=========================================="
    echo "Starting sampling for $EP at $(date)"
    echo "=========================================="
    python sample_phi4.py --method "em" --L 64 --device "cuda:2" --k "0.1923" --num_steps 2000 --ep "$EP"
    echo "Finished $EP at $(date)"
    echo ""
done

echo "All sampling jobs completed at $(date)"
