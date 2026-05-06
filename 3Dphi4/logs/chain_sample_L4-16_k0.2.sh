#!/bin/bash
# Wait for L4-16 k=0.2 training (PID $TRAIN_PID) to finish, then run L=64 cross-L sampling on cuda:2.
set -u
TRAIN_PID=1306016
cd /data/tyywork/DM/3Dphi4
source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

echo "[chain] waiting for train PID=$TRAIN_PID to exit ..."
tail --pid=$TRAIN_PID -f /dev/null
echo "[chain] train PID gone. checking final epoch ckpt ..."

RUN_DIR=runs/phi4_3d_Lmulti4-8-16_k0.2_l0.9_ncsnpp
EP=10000
if [ ! -f "$RUN_DIR/models/epoch=$(printf '%04d' $EP).ckpt" ]; then
    LATEST=$(ls "$RUN_DIR/models" | grep -oE 'epoch=[0-9]+' | sort -t= -k2 -n | tail -1 | cut -d= -f2)
    echo "[chain] epoch=$EP ckpt missing, falling back to latest: $LATEST"
    EP=$((10#$LATEST))
fi

echo "[chain] starting L=64 sampling with ep=$EP on cuda:2 ..."
python sample_phi4_crossL.py \
    --L_train 16 --L_sample 64 --k 0.2 --l 0.9 --ep $EP \
    --num_samples 128 --n_repeats 4 --num_steps 2000 \
    --method em --schedule log --seed 42 --device cuda:2 \
    --run_dir "$RUN_DIR"
echo "[chain] sampling done."
