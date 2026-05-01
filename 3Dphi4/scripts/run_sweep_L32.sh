#!/usr/bin/env bash
# Sweep sampling for 3D phi4 L=32, across 3 kappa values.
#
# Usage: ./run_sweep_L32.sh <gpu_idx> <kappa>
#   e.g. ./run_sweep_L32.sh 0 0.18
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

GPU=${1:?"usage: $0 <gpu_idx> <kappa>"}
KAPPA=${2:?"usage: $0 <gpu_idx> <kappa>"}
DEVICE="cuda:${GPU}"

L=32
LAMBDA=0.9
NETWORK=ncsnpp
STEP=5
NUM_SAMPLES=512
N_REPEATS=1
SDE_STEPS=2000
ODE_STEPS=400

RUN_DIR="runs/phi4_3d_L${L}_k${KAPPA}_l${LAMBDA}_${NETWORK}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="${LOG_DIR}/sweep_${DEVICE//:/}.log"

echo "==== SWEEP START $(date +%F\ %T)  device=${DEVICE}  L=${L}  κ=${KAPPA}  λ=${LAMBDA} ====" | tee -a "$SWEEP_LOG"

[[ -d "${RUN_DIR}/models" ]] || { echo "MISSING: ${RUN_DIR}/models" | tee -a "$SWEEP_LOG"; exit 1; }

mapfile -t CKPTS < <(ls "${RUN_DIR}/models/" | grep -E '^epoch=[0-9]+\.ckpt$' | sort -V)
n="${#CKPTS[@]}"
[[ $n -eq 0 ]] && { echo "empty models dir"; exit 1; }

SELECTED=()
for (( i=0; i<n; i+=STEP )); do SELECTED+=("${CKPTS[$i]}"); done
last="${CKPTS[$((n-1))]}"
[[ "${SELECTED[-1]}" != "$last" ]] && SELECTED+=("$last")

echo ">>> total=${n}  selected=${#SELECTED[@]}  epochs:" | tee -a "$SWEEP_LOG"
for c in "${SELECTED[@]}"; do echo "    ${c}"; done | tee -a "$SWEEP_LOG"

for CKPT in "${SELECTED[@]}"; do
    EP=$(basename "$CKPT" .ckpt | sed 's/epoch=//')
    CKPT_PATH="${RUN_DIR}/models/${CKPT}"
    SDE_OUT="${RUN_DIR}/data/samples_em_steps${SDE_STEPS}_${EP}.npy"
    ODE_OUT="${RUN_DIR}/data/samples_ode_steps${ODE_STEPS}_${EP}.npy"

    if [[ -f "$SDE_OUT" ]]; then
        echo "  [skip SDE] $SDE_OUT" | tee -a "$SWEEP_LOG"
    else
        echo "  [$(date +%T)] ep=${EP}  SDE em/${SDE_STEPS}" | tee -a "$SWEEP_LOG"
        python sample_phi4.py \
            --L "${L}" --k "${KAPPA}" --l "${LAMBDA}" \
            --checkpoint "${CKPT_PATH}" --ep "${EP}" \
            --method em --schedule log \
            --num_samples "${NUM_SAMPLES}" --num_steps "${SDE_STEPS}" \
            --n_repeats "${N_REPEATS}" \
            --device "${DEVICE}" --network "${NETWORK}" \
            >> "$SWEEP_LOG" 2>&1
    fi

    if [[ -f "$ODE_OUT" ]]; then
        echo "  [skip ODE] $ODE_OUT" | tee -a "$SWEEP_LOG"
    else
        echo "  [$(date +%T)] ep=${EP}  ODE dpm2/${ODE_STEPS}" | tee -a "$SWEEP_LOG"
        python sample_phi4.py \
            --L "${L}" --k "${KAPPA}" --l "${LAMBDA}" \
            --checkpoint "${CKPT_PATH}" --ep "${EP}" \
            --method ode --ode_method dpm2 --schedule log \
            --num_samples "${NUM_SAMPLES}" --num_steps "${ODE_STEPS}" \
            --n_repeats "${N_REPEATS}" \
            --device "${DEVICE}" --network "${NETWORK}" \
            >> "$SWEEP_LOG" 2>&1
    fi
done

echo "==== SWEEP END $(date +%F\ %T) ====" | tee -a "$SWEEP_LOG"
