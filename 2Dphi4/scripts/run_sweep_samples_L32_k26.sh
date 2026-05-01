#!/usr/bin/env bash
# Sweep sampling for L=32 k=0.26 sigma ablation.
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

GPU=${1:-3}
DEVICE="cuda:${GPU}"

case "${GPU}" in
    0) SIGMAS=(15 90)  ;;
    1) SIGMAS=(30 180) ;;
    3) SIGMAS=(60 360) ;;
    *) echo "gpu_idx must be 0, 1, or 3"; exit 1 ;;
esac

L=32
K=0.26
LAMBDA=0.022
NETWORK=ncsnpp
STEP=5
NUM_SAMPLES=512
N_REPEATS=1
SDE_STEPS=2000
ODE_STEPS=400

LOG_DIR="sigma_ablation_L32_k26_logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="${LOG_DIR}/sweep_${DEVICE//:/}_sigmas$(echo "${SIGMAS[@]}" | tr ' ' '-').log"

echo "==== SWEEP START $(date +%F\ %T) on ${DEVICE} (L=${L} k=${K}) ====" | tee -a "$SWEEP_LOG"

for SIGMA in "${SIGMAS[@]}"; do
    RUN_DIR="phi4_L${L}_k${K}_l${LAMBDA}_${NETWORK}_sigma${SIGMA}"
    [[ -d "${RUN_DIR}/models" ]] || { echo ">>> SKIP missing ${RUN_DIR}"; continue; }
    mapfile -t CKPTS < <(ls "${RUN_DIR}/models/" 2>/dev/null | grep -E '^epoch=[0-9]+\.ckpt$' | sort -V)
    n="${#CKPTS[@]}"; [[ $n -eq 0 ]] && continue

    SELECTED=()
    for (( i=0; i<n; i+=STEP )); do SELECTED+=("${CKPTS[$i]}"); done
    last="${CKPTS[$((n-1))]}"
    [[ "${SELECTED[-1]}" != "$last" ]] && SELECTED+=("$last")

    echo ">>> [$(date +%F\ %T)] sigma=${SIGMA}  selected=${#SELECTED[@]}" | tee -a "$SWEEP_LOG"

    for CKPT in "${SELECTED[@]}"; do
        EP=$(basename "$CKPT" .ckpt | sed 's/epoch=//')
        CKPT_PATH="${RUN_DIR}/models/${CKPT}"
        SDE_OUT="${RUN_DIR}/data/samples_em_steps${SDE_STEPS}_${EP}.npy"
        ODE_OUT="${RUN_DIR}/data/samples_ode_steps${ODE_STEPS}_${EP}.npy"

        if [[ ! -f "$SDE_OUT" ]]; then
            echo "    [$(date +%T)] sigma=${SIGMA} ep=${EP} SDE" | tee -a "$SWEEP_LOG"
            python sample_phi4.py --L "${L}" --k "${K}" --l "${LAMBDA}" \
                --checkpoint "${CKPT_PATH}" --ep "${EP}" --method em \
                --num_samples "${NUM_SAMPLES}" --num_steps "${SDE_STEPS}" \
                --n_repeats "${N_REPEATS}" --device "${DEVICE}" --network "${NETWORK}" \
                --output_suffix "_sigma${SIGMA}" >> "$SWEEP_LOG" 2>&1
        fi

        if [[ ! -f "$ODE_OUT" ]]; then
            echo "    [$(date +%T)] sigma=${SIGMA} ep=${EP} ODE" | tee -a "$SWEEP_LOG"
            python sample_phi4.py --L "${L}" --k "${K}" --l "${LAMBDA}" \
                --checkpoint "${CKPT_PATH}" --ep "${EP}" --method ode --ode_method dpm2 \
                --num_samples "${NUM_SAMPLES}" --num_steps "${ODE_STEPS}" \
                --n_repeats "${N_REPEATS}" --device "${DEVICE}" --network "${NETWORK}" \
                --output_suffix "_sigma${SIGMA}" >> "$SWEEP_LOG" 2>&1
        fi
    done
done
echo "==== SWEEP END $(date +%F\ %T) ====" | tee -a "$SWEEP_LOG"
