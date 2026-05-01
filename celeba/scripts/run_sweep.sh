#!/usr/bin/env bash
# Sweep all checkpoints of runs/celeba_64_ncsnpp with a given method/device.
# Usage: ./run_sweep.sh <device> <method> <num_steps> [epoch_list_file]
# Example: ./run_sweep.sh cuda:3 em 2000 epochs_all.txt
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

DEVICE=${1:?"device, e.g. cuda:3"}
METHOD=${2:?"method: em | dpm2 | ..."}
STEPS=${3:?"num_steps"}
EPOCH_FILE=${4:-epochs_all.txt}
SEED=${SEED:-20260417}
NUM_SAMPLES=${NUM_SAMPLES:-512}

LOG_DIR=runs/celeba_64_ncsnpp/logs
mkdir -p "$LOG_DIR"

while IFS= read -r EP; do
    [ -z "$EP" ] && continue
    echo ">>> $DEVICE $METHOD epoch=$EP"
    python sample_celeba.py \
        --ep "epoch=$EP" \
        --image_size 64 \
        --network ncsnpp \
        --method "$METHOD" \
        --num_steps "$STEPS" \
        --num_samples "$NUM_SAMPLES" \
        --n_repeats 1 \
        --schedule log \
        --seed "$SEED" \
        --device "$DEVICE" \
        2>&1 | tee -a "$LOG_DIR/sweep_${METHOD}_${DEVICE//:/}.log"
done < "$EPOCH_FILE"
