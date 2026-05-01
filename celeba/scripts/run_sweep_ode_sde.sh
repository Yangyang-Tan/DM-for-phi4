#!/usr/bin/env bash
# Parametric ODE+SDE sweep: 18 log-spaced epochs × {em (SDE), dpm2 (ODE)}.
#
# Usage: ./run_sweep_ode_sde.sh <L> <device>
#   L: 64 or 128
#   device: e.g. cuda:0
set -e

L=${1:?"L: 64 or 128"}
DEVICE=${2:?"device, e.g. cuda:0"}

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

NETWORK=ncsnpp
SDE_STEPS=2000
ODE_STEPS=400
SEED=20260421

RUN_DIR="celeba_${L}_${NETWORK}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR" "${RUN_DIR}/data"

# Per-L batch/repeat: target N=2048 total, constrained by GPU memory.
#   L=64  batch 2048 fits on 24GB 4090 (~6GB peak).
#   L=128 batch 2048 may OOM on 32GB 5090; split into 1024 × 2 repeats.
if [[ "$L" == "64" ]]; then
    NUM_SAMPLES=2048
    N_REPEATS=1
    EPOCHS=(0001 0002 0004 0006 0011 0017 0028 0045 0095 0151 0200 0351 0559 0890 1417 2257 3944 6280)
    EP_PREFIX=""
elif [[ "$L" == "128" ]]; then
    NUM_SAMPLES=1024
    N_REPEATS=2
    EPOCHS=(0001 0002 0004 0006 0011 0018 0029 0047 0099 0149 0199 0349 0549 0899 1449 2349 3799 6149)
    EP_PREFIX="epoch="
else
    echo "L must be 64 or 128"; exit 1
fi

LOG="${LOG_DIR}/sweep_ode_sde_${DEVICE//:/}.log"
echo "==== SWEEP START $(date +%F\ %T)  L=${L}  device=${DEVICE}  epochs=${#EPOCHS[@]} ====" | tee -a "$LOG"

for EP in "${EPOCHS[@]}"; do
    for spec in "em ${SDE_STEPS}" "dpm2 ${ODE_STEPS}"; do
        set -- $spec; METHOD=$1; STEPS=$2
        echo ">>> [$(date +%T)] L=${L} ep=${EP} ${METHOD}/${STEPS}" | tee -a "$LOG"
        python sample_celeba.py \
            --ep "${EP_PREFIX}epoch=${EP}" \
            --image_size "${L}" \
            --network "${NETWORK}" \
            --method "${METHOD}" \
            --num_steps "${STEPS}" \
            --num_samples "${NUM_SAMPLES}" \
            --n_repeats "${N_REPEATS}" \
            --schedule log \
            --seed "${SEED}" \
            --device "${DEVICE}" \
            >> "$LOG" 2>&1
    done
done

echo "==== SWEEP END $(date +%F\ %T) ====" | tee -a "$LOG"
