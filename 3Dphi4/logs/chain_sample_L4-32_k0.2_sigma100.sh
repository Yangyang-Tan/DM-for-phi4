#!/bin/bash
set -u
TRAIN_PID=3721107
DEV=cuda:3
K=0.2
RUN_DIR=runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp_sigma100

cd /data/tyywork/DM/3Dphi4
source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

echo "[chain] waiting for train PID=$TRAIN_PID to exit ..."
tail --pid=$TRAIN_PID -f /dev/null
echo "[chain] train PID gone."

EP=10000
if [ ! -f "$RUN_DIR/models/epoch=$(printf '%04d' $EP).ckpt" ]; then
    LATEST=$(ls "$RUN_DIR/models" | grep -oE 'epoch=[0-9]+' | sort -t= -k2 -n | tail -1 | cut -d= -f2)
    echo "[chain] epoch=$EP ckpt missing, falling back to latest: $LATEST"
    EP=$((10#$LATEST))
fi

for SCHED in linear log; do
    echo "[chain] [$SCHED] L=64 sampling ep=$EP on $DEV ..."
    python sample_phi4_crossL.py \
        --L_train 32 --L_sample 64 --k $K --l 0.9 --ep $EP \
        --num_samples 128 --n_repeats 4 --num_steps 2000 \
        --method em --schedule $SCHED --seed 42 --device $DEV \
        --run_dir "$RUN_DIR" || { echo "[chain] [$SCHED] FAILED"; continue; }
    echo "[chain] [$SCHED] done."
done
echo "[chain] all schedules done."
