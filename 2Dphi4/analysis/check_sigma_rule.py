"""
Empirical check: should the rule  σ_max ≈ 2·std(t=1)  be applied in
normalized [-1,1] data space, or in the original (un-normalized) data
space?

For each (k, L) with available training data:
  1. compute σ_max  (max pairwise Euclidean distance over a random
     subsample of training cfgs) in BOTH spaces;
  2. invert  std(t=1) = σ_max/2  to obtain the σ predicted by each rule;
  3. print a table next to the available ablation grid so the user can
     see which space's prediction matches their experimental best σ.

VE-SDE here:  std(t=1) = sqrt((σ²−1)/(2 ln σ))  for σ > 1.

CPU only; subsamples ``SAMPLE_N`` configs to keep pdist tractable.
"""

import math
from glob import glob

import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist


SAMPLE_N = 256          # configs subsampled for pdist
SEED = 0

ROOT = "/data/tyywork/DM/2Dphi4/trainingdata"

CASES = [
    # (k, L, ablation_grid)
    (0.26,    32, [15, 30, 60, 90, 180, 360]),
    (0.2705,  32, [15, 30, 60, 90, 180, 360]),
    (0.28,    32, [15, 30, 60, 90, 180, 360]),
    (0.26,    64, []),                          # no ablation grid trained
    (0.2705,  64, []),
    (0.28,    64, [25, 50, 100, 150, 300, 600]),
    # L=128 reference cases (already characterised, included for context)
    (0.2705, 128, [450]),
    (0.28,   128, [640]),
]


def std_t1(sigma: float) -> float:
    return math.sqrt((sigma ** 2 - 1.0) / (2.0 * math.log(sigma)))


def sigma_for_target_std(target: float) -> float:
    """Solve std_t1(σ) = target for σ ∈ (1.001, 1e6)."""
    f = lambda s: std_t1(s) - target
    return brentq(f, 1.001, 1.0e6)


def closest_in_grid(value: float, grid):
    if not grid:
        return None
    return min(grid, key=lambda g: abs(math.log(g) - math.log(value)))


def load_cfgs(k, L):
    pattern = f"{ROOT}/cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2"
    paths = glob(pattern)
    if not paths:
        return None
    with h5py.File(paths[0], "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    return cfgs


def normalize_pm1(arr, lo, hi):
    return ((arr - lo) / (hi - lo) - 0.5) * 2.0


def max_pairwise(x, n=SAMPLE_N, seed=SEED):
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    idx = rng.choice(N, size=min(n, N), replace=False)
    d = pdist(x[idx])
    return float(d.max()), float(d.mean())


def report():
    print("\n  σ_max measured on a random subsample of "
          f"{SAMPLE_N} cfgs;  std(t=1) = sqrt((σ²−1)/(2·ln σ)).\n")
    header = (f"{'k':>7} {'L':>4} | "
              f"{'σ_max^norm':>11} → {'σ_pred^norm':>11} | "
              f"{'σ_max^orig':>11} → {'σ_pred^orig':>11} | "
              f"{'closest_norm':>13} {'closest_orig':>13} | grid")
    print(header)
    print('-' * len(header))
    rows = []
    for k, L, grid in CASES:
        cfgs = load_cfgs(k, L)
        if cfgs is None:
            print(f"{k:>7} {L:>4} |   (training data missing)")
            continue
        flat_orig = cfgs.reshape(cfgs.shape[0], -1).astype(np.float32)
        nmin, nmax = float(flat_orig.min()), float(flat_orig.max())
        flat_norm = normalize_pm1(flat_orig, nmin, nmax)
        sm_n, _ = max_pairwise(flat_norm)
        sm_o, _ = max_pairwise(flat_orig)
        # Predicted σ from each rule: std(t=1) = σ_max / 2
        sigma_pred_n = sigma_for_target_std(sm_n / 2)
        sigma_pred_o = sigma_for_target_std(sm_o / 2)
        nearest_n = closest_in_grid(sigma_pred_n, grid)
        nearest_o = closest_in_grid(sigma_pred_o, grid)
        rows.append((k, L, sm_n, sigma_pred_n, sm_o, sigma_pred_o,
                     nearest_n, nearest_o, grid))
        print(f"{k:>7} {L:>4} | "
              f"{sm_n:11.3f} → {sigma_pred_n:11.2f} | "
              f"{sm_o:11.3f} → {sigma_pred_o:11.2f} | "
              f"{str(nearest_n):>13} {str(nearest_o):>13} | {grid}")

    print("\nInterpretation:")
    print("  σ_pred^norm  = σ predicted by  σ_max(normalized) ≈ 2·std(t=1)")
    print("  σ_pred^orig  = σ predicted by  σ_max(original)   ≈ 2·std(t=1)")
    print("  closest_*    = grid point of the ablation closest to σ_pred (log-distance)")
    print("\n  The 'right' space is the one whose σ_pred matches your best")
    print("  ablation σ across (k, L). Mismatch by ≫ one grid step ⇒ wrong space.\n")


if __name__ == "__main__":
    report()
