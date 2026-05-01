"""
Calibrate σ for multi-L joint training (κ=0.27088, L∈{8,16,32,64}).

Rule (CLAUDE.md, validated 2026-04 at L=128):
    σ_max ≈ 2·std(t=1)         in ORIGINAL (un-normalized) data space
    std(t=1) = sqrt((σ²−1)/(2 ln σ))

For multi-L joint training the network must cover the largest σ_max in
the pooled training set. σ_max grows with L (≈ √V scaling for cfgs with
roughly L-independent per-site std), so the binding L is L_max = 64.
We report σ_max(L) for every L plus the pooled σ_max so this is explicit.

We also print σ predictions in normalized [-1,1] space for completeness;
the validated rule lives in original space, so trust that column.
"""

import math
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist


import argparse
LAMBDA = 0.022
L_LIST_DEFAULT = [8, 16, 32, 64]
SAMPLE_N = 512   # for pdist
SEED = 0
ROOT = Path("/data/tyywork/DM/2Dphi4/trainingdata")


def std_t1(sigma: float) -> float:
    return math.sqrt((sigma ** 2 - 1.0) / (2.0 * math.log(sigma)))


def sigma_for_target_std(target: float) -> float:
    f = lambda s: std_t1(s) - target
    return brentq(f, 1.001, 1.0e6)


def max_pairwise(x, n=SAMPLE_N, seed=SEED):
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    idx = rng.choice(N, size=min(n, N), replace=False)
    d = pdist(x[idx])
    return float(d.max()), float(d.mean())


def load_cfgs(k, L):
    path = ROOT / f"cfgs_wolff_fahmc_k={k}_l={LAMBDA}_{L}^2.jld2"
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    sample_axis = int(np.argmax(cfgs.shape))
    if cfgs.ndim == 3 and sample_axis != 0:
        cfgs = np.moveaxis(cfgs, sample_axis, 0)
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.27088)
    p.add_argument("--Ls", type=str, default=",".join(str(x) for x in L_LIST_DEFAULT))
    args = p.parse_args()
    KAPPA = args.k
    L_LIST = [int(x) for x in args.Ls.split(",") if x.strip()]
    print(f"\nκ={KAPPA}, λ={LAMBDA}, L_list={L_LIST}")
    print(f"σ_max measured on a random subsample of N={SAMPLE_N} cfgs.\n")

    print(f"{'L':>5}  {'N':>7}  {'φ_min':>8} {'φ_max':>8}  "
          f"{'σ_max^orig':>11}  {'σ_pred^orig':>12}  "
          f"{'σ_max^norm':>11}  {'σ_pred^norm':>12}")
    print("-" * 92)

    raws = {}
    sigma_max_orig = {}
    sigma_max_norm = {}
    for L in L_LIST:
        cfgs = load_cfgs(KAPPA, L)
        flat_orig = cfgs.reshape(cfgs.shape[0], -1).astype(np.float32)
        raws[L] = flat_orig

        sm_o, _ = max_pairwise(flat_orig)
        sigma_max_orig[L] = sm_o
        sigma_pred_o = sigma_for_target_std(sm_o / 2)

        # Normalised within-L (just for the per-L diagnostic column)
        lo_L, hi_L = float(flat_orig.min()), float(flat_orig.max())
        flat_norm = ((flat_orig - lo_L) / (hi_L - lo_L) - 0.5) * 2.0
        sm_n, _ = max_pairwise(flat_norm)
        sigma_max_norm[L] = sm_n
        sigma_pred_n = sigma_for_target_std(sm_n / 2)

        print(f"{L:>5}  {cfgs.shape[0]:>7}  {lo_L:>8.3f} {hi_L:>8.3f}  "
              f"{sm_o:>11.3f}  {sigma_pred_o:>12.2f}  "
              f"{sm_n:>11.3f}  {sigma_pred_n:>12.2f}")

    # Pooled (multi-L) — original space: simply the max over L.
    pooled_orig = max(sigma_max_orig.values())
    sigma_pool_orig = sigma_for_target_std(pooled_orig / 2)

    # Pooled normalisation as the multi-L pipeline does it: single global
    # (lo, hi) over all L's; σ_max in that pooled-norm space.
    pool_lo = min(r.min() for r in raws.values())
    pool_hi = max(r.max() for r in raws.values())
    pool_norm_per_L = []
    for L in L_LIST:
        nrm = ((raws[L] - pool_lo) / (pool_hi - pool_lo) - 0.5) * 2.0
        sm_pn, _ = max_pairwise(nrm)
        pool_norm_per_L.append((L, sm_pn))
    pooled_norm = max(s for _, s in pool_norm_per_L)
    sigma_pool_norm = sigma_for_target_std(pooled_norm / 2)

    print("\nPooled (multi-L joint training):")
    print(f"  σ_max^orig       = {pooled_orig:.3f}  (binding L = "
          f"{max(sigma_max_orig, key=sigma_max_orig.get)})")
    print(f"  σ_pred^orig      = {sigma_pool_orig:.2f}     ← USE THIS "
          f"(rule validated at L=128 in original space)")
    print(f"  σ_max^pool-norm  = {pooled_norm:.3f}")
    print(f"  σ_pred^pool-norm = {sigma_pool_norm:.2f}     (alt; only relevant "
          f"if rule turned out to live in normalized space)")

    # Reference: single-L predictions for the same κ, useful for sanity-check.
    print("\nReference — single-L σ predictions in original space:")
    for L in L_LIST:
        s = sigma_for_target_std(sigma_max_orig[L] / 2)
        print(f"  L={L:>3}: σ_max={sigma_max_orig[L]:.2f} → σ ≈ {s:.1f}")

    # Comparison with the existing κ=0.2705 multi-L run (σ=360 was used).
    print("\nFor reference, prior κ=0.2705 multi-L L∈{8,16,32,64} used σ=360.")


if __name__ == "__main__":
    main()
