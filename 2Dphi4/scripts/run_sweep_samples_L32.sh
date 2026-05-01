#!/usr/bin/env bash
# Sweep sampling for L=32 sigma ablation.
# Every 5th log-scale ckpt (plus the last), 512 samples, SDE (em 2000) + ODE (dpm2 400).
#
# Usage:
#   ./run_sweep_samples_L32.sh <gpu_idx>    # gpu_idx ∈ {0, 1, 3}
#                                           # omit for single-GPU default (cuda:3, all sigmas)
#
# 3-GPU split (sweep two sigmas per GPU):
#   cuda:0 -> σ=15, 90
#   cuda:1 -> σ=30, 180
#   cuda:3 -> σ=60, 360
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

GPU=${1:-3}
DEVICE="cuda:${GPU}"

case "${GPU}" in
    0) SIGMAS=(15 90)    ;;
    1) SIGMAS=(30 180)   ;;
    3) SIGMAS=(60 360)   ;;
    all) SIGMAS=(15 30 60 90 180 360); DEVICE="cuda:3" ;;
    *) echo "gpu_idx must be 0, 1, 3, or 'all'"; exit 1 ;;
esac

L=32
K=0.28
LAMBDA=0.022
NETWORK=ncsnpp
STEP=5
NUM_SAMPLES=512
N_REPEATS=1
SDE_STEPS=2000
ODE_STEPS=400

LOG_DIR="results/sigma_ablation/L32"
mkdir -p "$LOG_DIR"
SWEEP_LOG="${LOG_DIR}/sweep_${DEVICE//:/}_sigmas$(echo "${SIGMAS[@]}" | tr ' ' '-').log"

echo "==== SWEEP START $(date +%F\ %T) on ${DEVICE} (L=${L}) ====" | tee -a "$SWEEP_LOG"

for SIGMA in "${SIGMAS[@]}"; do
    RUN_DIR="runs/phi4_L${L}_k${K}_l${LAMBDA}_${NETWORK}_sigma${SIGMA}"
    if [[ ! -d "${RUN_DIR}/models" ]]; then
        echo ">>> [SKIP] ${RUN_DIR}/models missing" | tee -a "$SWEEP_LOG"
        continue
    fi

    mapfile -t CKPTS < <(ls "${RUN_DIR}/models/" 2>/dev/null | grep -E '^epoch=[0-9]+\.ckpt$' | sort -V)
    n="${#CKPTS[@]}"
    if (( n == 0 )); then
        echo ">>> [SKIP] ${RUN_DIR}/models empty" | tee -a "$SWEEP_LOG"
        continue
    fi

    SELECTED=()
    for (( i=0; i<n; i+=STEP )); do
        SELECTED+=("${CKPTS[$i]}")
    done
    last="${CKPTS[$((n-1))]}"
    if [[ "${SELECTED[-1]}" != "$last" ]]; then
        SELECTED+=("$last")
    fi

    echo ">>> [$(date +%F\ %T)] sigma=${SIGMA}  total_ckpts=${n}  selected=${#SELECTED[@]}" | tee -a "$SWEEP_LOG"

    for CKPT in "${SELECTED[@]}"; do
        EP=$(basename "$CKPT" .ckpt | sed 's/epoch=//')
        CKPT_PATH="${RUN_DIR}/models/${CKPT}"
        SDE_OUT="${RUN_DIR}/data/samples_em_steps${SDE_STEPS}_${EP}.npy"
        ODE_OUT="${RUN_DIR}/data/samples_ode_steps${ODE_STEPS}_${EP}.npy"

        if [[ -f "$SDE_OUT" ]]; then
            echo "    [skip SDE] exists: $SDE_OUT" | tee -a "$SWEEP_LOG"
        else
            echo "    [$(date +%T)] sigma=${SIGMA}  ep=${EP}  SDE em/${SDE_STEPS}" | tee -a "$SWEEP_LOG"
            python sample_phi4.py \
                --L "${L}" --k "${K}" --l "${LAMBDA}" \
                --checkpoint "${CKPT_PATH}" \
                --ep "${EP}" \
                --method em \
                --num_samples "${NUM_SAMPLES}" \
                --num_steps "${SDE_STEPS}" \
                --n_repeats "${N_REPEATS}" \
                --device "${DEVICE}" \
                --network "${NETWORK}" \
                --output_suffix "_sigma${SIGMA}" \
                >> "$SWEEP_LOG" 2>&1
        fi

        if [[ -f "$ODE_OUT" ]]; then
            echo "    [skip ODE] exists: $ODE_OUT" | tee -a "$SWEEP_LOG"
        else
            echo "    [$(date +%T)] sigma=${SIGMA}  ep=${EP}  ODE dpm2/${ODE_STEPS}" | tee -a "$SWEEP_LOG"
            python sample_phi4.py \
                --L "${L}" --k "${K}" --l "${LAMBDA}" \
                --checkpoint "${CKPT_PATH}" \
                --ep "${EP}" \
                --method ode --ode_method dpm2 \
                --num_samples "${NUM_SAMPLES}" \
                --num_steps "${ODE_STEPS}" \
                --n_repeats "${N_REPEATS}" \
                --device "${DEVICE}" \
                --network "${NETWORK}" \
                --output_suffix "_sigma${SIGMA}" \
                >> "$SWEEP_LOG" 2>&1
        fi
    done

    echo ">>> [$(date +%F\ %T)] sigma=${SIGMA}  DONE" | tee -a "$SWEEP_LOG"
done

echo "==== SWEEP END $(date +%F\ %T) ====" | tee -a "$SWEEP_LOG"
