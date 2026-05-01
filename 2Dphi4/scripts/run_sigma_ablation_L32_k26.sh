#!/usr/bin/env bash
# Sigma ablation for 2Dphi4 L=32 k=0.26 l=0.022 (symmetric phase, D_max=14.11).
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

GPU=${1:?"usage: $0 <gpu_idx>   (0 | 1 | 3)"}
DEVICE="cuda:${GPU}"

case "${GPU}" in
    0) SIGMAS=(15 90)   ;;
    1) SIGMAS=(30 180)  ;;
    3) SIGMAS=(60 360)  ;;
    *) echo "gpu_idx must be 0, 1, or 3"; exit 1 ;;
esac

L=32
K=0.26
LAMBDA=0.022
EPOCHS=10000
BATCH=256
NUM_CKPTS=100
NETWORK=ncsnpp
DATA_PATH="trainingdata/cfgs_wolff_fahmc_k=${K}_l=${LAMBDA}_${L}^2.jld2"

LOG_DIR="results/sigma_ablation/L32_k26"
mkdir -p "$LOG_DIR"

for SIGMA in "${SIGMAS[@]}"; do
    SUFFIX="_sigma${SIGMA}"
    RUN_DIR="runs/phi4_L${L}_k${K}_l${LAMBDA}_${NETWORK}${SUFFIX}"
    LOG="${LOG_DIR}/train_L${L}_k${K}${SUFFIX}_${DEVICE//:/}.log"

    RESUME_FLAG=""
    if [[ -d "${RUN_DIR}/models" ]]; then
        LAST_CKPT=$(ls "${RUN_DIR}/models/" 2>/dev/null | grep -E '^epoch=[0-9]+\.ckpt$' | sort -V | tail -1)
        if [[ -n "$LAST_CKPT" ]]; then
            RESUME_FLAG="--ckpt_path ${RUN_DIR}/models/${LAST_CKPT}"
            echo ">>> [$(date +%F\ %T)] ${DEVICE}  RESUME  sigma=${SIGMA}  from ${LAST_CKPT}"
        fi
    fi
    [[ -z "$RESUME_FLAG" ]] && echo ">>> [$(date +%F\ %T)] ${DEVICE}  TRAIN  sigma=${SIGMA}  (fresh)"

    python train_phi4.py \
        --L "${L}" --k "${K}" --l "${LAMBDA}" \
        --sigma "${SIGMA}" \
        --batch_size "${BATCH}" --epochs "${EPOCHS}" \
        --num_ckpts "${NUM_CKPTS}" \
        --device "${DEVICE}" \
        --data_path "${DATA_PATH}" \
        --gpu_data --network "${NETWORK}" \
        --output_suffix "${SUFFIX}" \
        ${RESUME_FLAG} \
        2>&1 | tee -a "${LOG}"
done

echo ">>> [$(date +%F\ %T)] ${DEVICE} all trainings done (k=${K})"
