#!/usr/bin/env bash
# Sweep sampling for a single (k, sigma) at L=128, 2048 samples per run,
# 18 log-spaced epochs × (EM-SDE/2000 + DPM2-ODE/400).
#
# Usage:
#   ./run_sweep_samples_L128.sh <k> <sigma> <device>
# Example:
#   ./run_sweep_samples_L128.sh 0.28   640 cuda:2
#   ./run_sweep_samples_L128.sh 0.2705 450 cuda:3
set -e

K=${1:?"k, e.g. 0.28 or 0.2705"}
SIGMA=${2:?"sigma suffix number, e.g. 640 or 450"}
DEVICE=${3:?"device, e.g. cuda:2"}

L=128
LAMBDA=0.022
NETWORK=ncsnpp
# batch 2048 single-shot OOMs on 32GB 5090 during dpm2 (2nd-order ODE keeps
# 2 scores + intermediate x in memory); split into 1024 × 2 repeats.
# With --seed s, rep0 uses seed=s, rep1 uses seed=s+1 → deterministic + shared
# across epochs, so same-IC cross-epoch comparison still holds.
NUM_SAMPLES=1024
N_REPEATS=2
SDE_STEPS=2000
ODE_STEPS=400
SEED=20260422

RUN_DIR="runs/phi4_L${L}_k${K}_l${LAMBDA}_${NETWORK}_sigma${SIGMA}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR" "${RUN_DIR}/data"

# 18 log-spaced epochs chosen from available log-scale checkpoints [1..10000]
EPOCHS=(0001 0002 0003 0005 0009 0016 0028 0045 0079 0138 0242 0422 0739 1291 2257 3593 6280 10000)

LOG="${LOG_DIR}/sweep_L${L}_k${K}_sigma${SIGMA}_${DEVICE//:/}.log"
echo "==== SWEEP START $(date +%F\ %T)  L=${L} k=${K} sigma=${SIGMA} device=${DEVICE} ====" | tee -a "$LOG"

for EP_NUM in "${EPOCHS[@]}"; do
    EP="epoch=${EP_NUM}"
    CKPT="${RUN_DIR}/models/${EP}.ckpt"
    if [[ ! -f $CKPT ]]; then
        echo ">>> [SKIP missing] $CKPT" | tee -a "$LOG"
        continue
    fi

    # 1) SDE Euler-Maruyama
    echo ">>> [$(date +%T)] k=${K} sigma=${SIGMA} ep=${EP_NUM}  em/${SDE_STEPS}" | tee -a "$LOG"
    conda run --no-capture-output -n nenv2 python sample_phi4.py \
        --L ${L} --k ${K} --l ${LAMBDA} \
        --network ${NETWORK} --output_suffix "_sigma${SIGMA}" \
        --ep "${EP}" \
        --method em --num_steps ${SDE_STEPS} \
        --num_samples ${NUM_SAMPLES} --n_repeats ${N_REPEATS} \
        --seed ${SEED} \
        --device ${DEVICE} \
        >> "$LOG" 2>&1

    # 2) ODE DPM-Solver 2
    echo ">>> [$(date +%T)] k=${K} sigma=${SIGMA} ep=${EP_NUM}  ode/dpm2/${ODE_STEPS}" | tee -a "$LOG"
    conda run --no-capture-output -n nenv2 python sample_phi4.py \
        --L ${L} --k ${K} --l ${LAMBDA} \
        --network ${NETWORK} --output_suffix "_sigma${SIGMA}" \
        --ep "${EP}" \
        --method ode --ode_method dpm2 --num_steps ${ODE_STEPS} \
        --num_samples ${NUM_SAMPLES} --n_repeats ${N_REPEATS} \
        --seed ${SEED} \
        --device ${DEVICE} \
        >> "$LOG" 2>&1
done

echo "==== SWEEP END $(date +%F\ %T) ====" | tee -a "$LOG"
