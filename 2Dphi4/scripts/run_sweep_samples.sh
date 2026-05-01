#!/usr/bin/env bash
# Sweep sampling across log-scale checkpoints for the sigma ablation.
#
# For each sigma in SIGMAS, select every Nth checkpoint (N=STEP) from the sorted
# log-scale ckpt list (always include the last one), and run BOTH:
#   - SDE (EM, --num_steps 2000)
#   - ODE (DPM-2, --num_steps 400)
# with 512 samples each.
#
# Sigmas finished first are processed first (25, 50, 100, 600). Sigmas still
# training (150, 300) are processed last — by which time they've likely finished.
#
# Usage:
#   ./run_sweep_samples.sh [device]
#   DEVICE defaults to cuda:3.
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

DEVICE=${1:-cuda:3}

L=64
K=0.28
LAMBDA=0.022
NETWORK=ncsnpp
STEP=5
NUM_SAMPLES=512
N_REPEATS=1
SDE_STEPS=2000
ODE_STEPS=400

# Order: completed sigmas first, still-training last
SIGMAS=(25 50 100 600 150 300)

LOG_DIR="results/sigma_ablation/all"
mkdir -p "$LOG_DIR"
SWEEP_LOG="${LOG_DIR}/sweep_${DEVICE//:/}.log"

echo "==== SWEEP START $(date +%F\ %T) on ${DEVICE} ====" | tee -a "$SWEEP_LOG"

for SIGMA in "${SIGMAS[@]}"; do
    RUN_DIR="runs/phi4_L64_k${K}_l${LAMBDA}_${NETWORK}_sigma${SIGMA}"
    if [[ ! -d "${RUN_DIR}/models" ]]; then
        echo ">>> [SKIP] ${RUN_DIR}/models missing" | tee -a "$SWEEP_LOG"
        continue
    fi

    # Snapshot current ckpt list (version-sorted by epoch number)
    mapfile -t CKPTS < <(ls "${RUN_DIR}/models/" 2>/dev/null | grep -E '^epoch=[0-9]+\.ckpt$' | sort -V)
    n="${#CKPTS[@]}"
    if (( n == 0 )); then
        echo ">>> [SKIP] ${RUN_DIR}/models empty" | tee -a "$SWEEP_LOG"
        continue
    fi

    # Select every STEP-th, always include last
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

        # SDE
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

        # ODE
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
