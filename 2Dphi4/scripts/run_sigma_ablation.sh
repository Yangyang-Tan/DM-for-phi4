#!/usr/bin/env bash
# Sigma ablation for 2Dphi4 L=64 k=0.28 l=0.022.
#
# Splits 6 sigma values across 3 GPUs (cuda:0, cuda:1, cuda:3).
# Each GPU runs 2 sigmas sequentially, and for each sigma:
#   1) train 10000 epochs
#   2) sample with SDE (em, 2000 steps)
#   3) sample with ODE (dpm2, 400 steps)
#
# Usage:
#   ./run_sigma_ablation.sh <gpu_idx>        # gpu_idx ∈ {0, 1, 3}
#
# Example (launch three terminals / tmux panes):
#   ./run_sigma_ablation.sh 0   # sigma=25, then 150 on cuda:0
#   ./run_sigma_ablation.sh 1   # sigma=50, then 300 on cuda:1
#   ./run_sigma_ablation.sh 3   # sigma=100, then 600 on cuda:3
set -e

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate nenv2

GPU=${1:?"usage: $0 <gpu_idx>   (0 | 1 | 3)"}
DEVICE="cuda:${GPU}"

# Per-GPU sigma list
case "${GPU}" in
    0) SIGMAS=(25  150) ;;
    1) SIGMAS=(50  300) ;;
    3) SIGMAS=(100 600) ;;
    *) echo "gpu_idx must be 0, 1, or 3"; exit 1 ;;
esac

L=64
K=0.28
LAMBDA=0.022
EPOCHS=10000
BATCH=256
NUM_CKPTS=100
DATA_PATH="trainingdata/cfgs_wolff_fahmc_k=${K}_l=${LAMBDA}_${L}^2.jld2"

# Sampling settings
SDE_STEPS=2000
ODE_STEPS=400
NUM_SAMPLES_PER_REP=1024
N_REPEATS=4
SAMPLE_EP=$EPOCHS   # use the final log-scale checkpoint (epoch=10000.ckpt)

LOG_DIR="results/sigma_ablation/all"
mkdir -p "$LOG_DIR"

for SIGMA in "${SIGMAS[@]}"; do
    SUFFIX="_sigma${SIGMA}"
    TAG="L${L}_k${K}${SUFFIX}_${DEVICE//:/}"

    # ----------------- train -----------------
    TRAIN_LOG="${LOG_DIR}/train_${TAG}.log"
    echo ">>> [$(date +%F\ %T)] ${DEVICE}  TRAIN  sigma=${SIGMA}"
    python train_phi4.py \
        --L "${L}" --k "${K}" --l "${LAMBDA}" \
        --sigma "${SIGMA}" \
        --batch_size "${BATCH}" --epochs "${EPOCHS}" \
        --num_ckpts "${NUM_CKPTS}" \
        --device "${DEVICE}" \
        --data_path "${DATA_PATH}" \
        --gpu_data \
        --network ncsnpp \
        --output_suffix "${SUFFIX}" \
        2>&1 | tee "${TRAIN_LOG}"

    # ----------------- sample SDE (EM, 2000 steps) -----------------
    SDE_LOG="${LOG_DIR}/sample_sde_${TAG}.log"
    echo ">>> [$(date +%F\ %T)] ${DEVICE}  SAMPLE-SDE  sigma=${SIGMA}  steps=${SDE_STEPS}"
    python sample_phi4.py \
        --L "${L}" --k "${K}" --l "${LAMBDA}" \
        --ep "${SAMPLE_EP}" \
        --method em \
        --num_samples "${NUM_SAMPLES_PER_REP}" \
        --num_steps "${SDE_STEPS}" \
        --n_repeats "${N_REPEATS}" \
        --device "${DEVICE}" \
        --network ncsnpp \
        --output_suffix "${SUFFIX}" \
        2>&1 | tee "${SDE_LOG}"

    # ----------------- sample ODE (DPM-2, 400 steps) -----------------
    ODE_LOG="${LOG_DIR}/sample_ode_${TAG}.log"
    echo ">>> [$(date +%F\ %T)] ${DEVICE}  SAMPLE-ODE  sigma=${SIGMA}  steps=${ODE_STEPS}"
    python sample_phi4.py \
        --L "${L}" --k "${K}" --l "${LAMBDA}" \
        --ep "${SAMPLE_EP}" \
        --method ode --ode_method dpm2 \
        --num_samples "${NUM_SAMPLES_PER_REP}" \
        --num_steps "${ODE_STEPS}" \
        --n_repeats "${N_REPEATS}" \
        --device "${DEVICE}" \
        --network ncsnpp \
        --output_suffix "${SUFFIX}" \
        2>&1 | tee "${ODE_LOG}"

    echo ">>> [$(date +%F\ %T)] ${DEVICE}  DONE  sigma=${SIGMA}"
done

echo ">>> [$(date +%F\ %T)] ${DEVICE} ALL SIGMAS DONE"
