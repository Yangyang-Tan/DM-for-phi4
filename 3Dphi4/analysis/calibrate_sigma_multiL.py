"""
Calibrate σ for 3D multi-L joint training (κ=0.1923, L∈{4,8,16,32}).

Rule (3D, validated against historical L=64 σ choices in this repo):
    std(t=1) ≈ σ_max         in ORIGINAL (un-normalized) data space
    std(t=1) = sqrt((σ²−1)/(2 ln σ))

Note: this differs from the 2D rule (σ_max ≈ 2·std(t=1)) by a factor of 2.
Empirical cross-check:
  - L=64 k=0.2:    σ_max ≈ 767, used σ=2760 → std(t=1)=693 ≈ σ_max ✓
  - L=64 k=0.1923: used σ=2048, predicted σ_max ≈ 524 (consistent)

In multi-L joint training the network must cover the largest σ_max in the
pooled training set. σ_max grows with L (≈ √V scaling for cfgs with roughly
L-independent per-site std), so the binding L is L_max.
"""

import math
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist


LAMBDA = 0.9
L_LIST_DEFAULT = [4, 8, 16, 32]
SAMPLE_N = 256   # for pdist; 3D cfgs are large so keep moderate
SEED = 0
ROOT = Path("/data/tyywork/DM/3Dphi4/trainingdata")


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
    path = ROOT / f"cfgs_wolff_fahmc_k={k}_l={LAMBDA}_{L}^3.jld2"
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    # Auto-detect (L,L,L,N) -> (N,L,L,L)
    if cfgs.ndim == 4 and cfgs.shape[-1] > cfgs.shape[0]:
        cfgs = cfgs.transpose(3, 0, 1, 2)
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--Ls", type=str, default=",".join(str(x) for x in L_LIST_DEFAULT))
    args = p.parse_args()
    KAPPA = args.k
    L_LIST = [int(x) for x in args.Ls.split(",") if x.strip()]
    print(f"\nκ={KAPPA}, λ={LAMBDA}, L_list={L_LIST}  (3D)")
    print(f"σ_max measured on a random subsample of N={SAMPLE_N} cfgs.\n")

    print(f"{'L':>5}  {'N':>7}  {'φ_min':>8} {'φ_max':>8}  "
          f"{'σ_max^orig':>11}  {'σ_pred^orig':>12}  "
          f"{'σ_max^norm':>11}  {'σ_pred^norm':>12}")
    print("-" * 92)

    raws = {}
    sigma_max_orig = {}
    for L in L_LIST:
        cfgs = load_cfgs(KAPPA, L)
        flat_orig = cfgs.reshape(cfgs.shape[0], -1).astype(np.float32)
        raws[L] = flat_orig

        sm_o, _ = max_pairwise(flat_orig)
        sigma_max_orig[L] = sm_o
        sigma_pred_o = sigma_for_target_std(sm_o)

        lo_L, hi_L = float(flat_orig.min()), float(flat_orig.max())
        flat_norm = ((flat_orig - lo_L) / (hi_L - lo_L) - 0.5) * 2.0
        sm_n, _ = max_pairwise(flat_norm)
        sigma_pred_n = sigma_for_target_std(sm_n)

        print(f"{L:>5}  {cfgs.shape[0]:>7}  {lo_L:>8.3f} {hi_L:>8.3f}  "
              f"{sm_o:>11.3f}  {sigma_pred_o:>12.2f}  "
              f"{sm_n:>11.3f}  {sigma_pred_n:>12.2f}")

    pooled_orig = max(sigma_max_orig.values())
    sigma_pool_orig = sigma_for_target_std(pooled_orig)

    pool_lo = min(r.min() for r in raws.values())
    pool_hi = max(r.max() for r in raws.values())
    pooled_norm = 0.0
    for L in L_LIST:
        nrm = ((raws[L] - pool_lo) / (pool_hi - pool_lo) - 0.5) * 2.0
        sm_pn, _ = max_pairwise(nrm)
        pooled_norm = max(pooled_norm, sm_pn)
    sigma_pool_norm = sigma_for_target_std(pooled_norm)

    binding_L = max(sigma_max_orig, key=sigma_max_orig.get)
    print("\nPooled (multi-L joint training):")
    print(f"  σ_max^orig       = {pooled_orig:.3f}  (binding L = {binding_L})")
    print(f"  σ_pred^orig      = {sigma_pool_orig:.2f}     ← USE THIS")
    print(f"  σ_max^pool-norm  = {pooled_norm:.3f}")
    print(f"  σ_pred^pool-norm = {sigma_pool_norm:.2f}")

    print("\nReference — single-L σ predictions in original space:")
    for L in L_LIST:
        s = sigma_for_target_std(sigma_max_orig[L])
        print(f"  L={L:>3}: σ_max={sigma_max_orig[L]:.2f} → σ ≈ {s:.1f}")


if __name__ == "__main__":
    main()
