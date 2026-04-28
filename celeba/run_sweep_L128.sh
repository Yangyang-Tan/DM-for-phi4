#!/usr/bin/env bash
# Sweep sampling of CelebA L=128: 19 log-scale epochs × (SDE em/2000 + ODE dpm2/400)
# 512 samples per epoch.
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

DEVICE=${1:-cuda:2}
IMAGE_SIZE=128
NETWORK=ncsnpp
NUM_SAMPLES=512
N_REPEATS=1
SDE_STEPS=2000
ODE_STEPS=400
SEED=20260421

RUN_DIR="celeba_${IMAGE_SIZE}_${NETWORK}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR" "${RUN_DIR}/data"

# 19 log-scale epochs chosen from available ckpts (1..9999)
EPOCHS=(1 2 4 6 11 18 29 47 99 149 199 349 549 899 1449 2349 3799 6149 9999)

LOG="${LOG_DIR}/sweep_L128_${DEVICE//:/}.log"
echo "==== SWEEP START $(date +%F\ %T)  device=${DEVICE}  L=${IMAGE_SIZE}  epochs=${#EPOCHS[@]} ====" | tee -a "$LOG"

for EP_NUM in "${EPOCHS[@]}"; do
    EP=$(printf "%04d" "$EP_NUM")
    CKPT="${RUN_DIR}/models/epoch=epoch=${EP}.ckpt"
    if [[ ! -f "$CKPT" ]]; then
        echo ">>> [SKIP missing] $CKPT" | tee -a "$LOG"
        continue
    fi

    for spec in "em ${SDE_STEPS}" "dpm2 ${ODE_STEPS}"; do
        set -- $spec; METHOD=$1; STEPS=$2
        # sample_celeba.py save naming convention: check after first run
        echo ">>> [$(date +%T)] ep=${EP}  ${METHOD}/${STEPS}  (seed=${SEED})" | tee -a "$LOG"
        python sample_celeba.py \
            --checkpoint "${CKPT}" \
            --ep "epoch=${EP}" \
            --image_size "${IMAGE_SIZE}" \
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
