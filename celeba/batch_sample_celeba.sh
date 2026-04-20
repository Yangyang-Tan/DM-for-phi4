#!/bin/bash
# Batch sample from CelebA diffusion model at selected epochs.
# For studying propagator evolution across the full training process.
#
# Usage:
#   bash batch_sample_celeba.sh
#
# Each epoch: 256 samples, 2000 steps (log), ~94s per epoch
# Total: ~30 epochs × 94s ≈ 47 minutes

DEVICE="cuda:3"
NETWORK="ncsnpp"
NUM_SAMPLES=512
NUM_STEPS=4000
SCHEDULE="log"
N_REPEATS=1

# Epoch selection:
#   Early (dense):  0, 2, 5, 10, 20, 30, 47
#   Mid (convergence):  99, 199, 299, 499, 699, 999
#   Late (plateau, dense for sliding window):
#     1099, 1199, 1299, 1399, 1499,
#     1599, 1699, 1799, 1899, 1999,
#     2099, 2199, 2299, 2399, 2499, 2649
EPOCHS=(
    # "epoch=0000" "epoch=0002" "epoch=0005" "epoch=0010"
    # "epoch=0020" "epoch=0030" "epoch=0047"
    # "epoch=0099" "epoch=0199" "epoch=0299" "epoch=0499"
    # "epoch=0699" "epoch=0999"
    # "epoch=1099" "epoch=1199" "epoch=1299" "epoch=1399" "epoch=1499"
    "epoch=1599" "epoch=1699" "epoch=1799" "epoch=1899" "epoch=1999"
    "epoch=2099" "epoch=2199" "epoch=2299" "epoch=2399" "epoch=2499" "epoch=2649"
)

TOTAL=${#EPOCHS[@]}
COUNT=0

for EP in "${EPOCHS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=============================================="
    echo "  [$COUNT/$TOTAL] Sampling: $EP"
    echo "=============================================="
    python sample_celeba.py \
        --ep "$EP" \
        --network "$NETWORK" \
        --device "$DEVICE" \
        --num_samples "$NUM_SAMPLES" \
        --num_steps "$NUM_STEPS" \
        --schedule "$SCHEDULE" \
        --n_repeats "$N_REPEATS"
done

echo ""
echo "All $TOTAL epochs sampled!"
