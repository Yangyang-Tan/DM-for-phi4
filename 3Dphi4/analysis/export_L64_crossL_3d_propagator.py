"""
Export the diagonal propagator G(|k_lat|) for the 3D L=64 cross-L comparison
(k=0.1923, lambda=0.9) into a long-format CSV for Mathematica re-plotting.

Sources mirror compare_L64_crossL_3d_all.py, except L=16-only uses ep=9111
(the only checkpoint that has the crossL64 sample on disk).
"""
from __future__ import annotations
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)


def main():
    k = 0.1923
    lam = 0.9
    L_sample = 64

    cases = [
        ("HMC_L64",          None,
         f"trainingdata/cfgs_wolff_fahmc_k={k}_l={lam}_{L_sample}^3.jld2"),
        ("L4_only",           9929,
         f"runs/phi4_3d_L4_k{k}_l{lam}_ncsnpp_sigma40/data_crossL/"
         f"samples_crossL64_L4only_ep9929_em_linear_steps2000_512.npy"),
        ("L8_only",          10000,
         f"runs/phi4_3d_L8_k{k}_l{lam}_ncsnpp_sigma100/data_crossL/"
         f"samples_crossL64_L8only_ep10000_em_linear_steps2000_512.npy"),
        ("multiL_4-8",       10000,
         f"runs/phi4_3d_Lmulti4-8_k{k}_l{lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_multiL4-8_ep10000_em_linear_steps2000_512.npy"),
        ("L16_only",          9111,
         f"runs/phi4_3d_L16_k{k}_l{lam}_ncsnpp_sigma280/data_crossL/"
         f"samples_crossL64_L16only_ep9111_em_linear_steps2000_512.npy"),
        ("multiL_4-8-16",    10000,
         f"runs/phi4_3d_Lmulti4-8-16_k{k}_l{lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_multiL4-8-16_ep10000_em_linear_steps2000_512.npy"),
        ("L32_only",         10000,
         f"runs/phi4_3d_L32_k{k}_l{lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_L32only_ep10000_em_linear_steps2000_512.npy"),
        ("multiL_4-8-16-32", 10000,
         f"runs/phi4_3d_Lmulti4-8-16-32_k{k}_l{lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_multiL_ep10000_em_linear_steps2000_512.npy"),
    ]

    rows = []
    print(f"{'method':>22}  {'ep':>5}  {'N':>5}  {'G(k_min)':>16}")
    print("-" * 60)
    for name, ep, path in cases:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name}: {p}")
            continue
        cfgs = load_hmc_3d(p) if name.startswith("HMC") else load_dm_3d(p)
        k_lat, G, dG = diagonal_propagator_3d(cfgs)
        N = cfgs.shape[0]
        ep_str = "" if ep is None else str(ep)
        for kk, gg, ee in zip(k_lat, G, dG):
            rows.append((name, ep_str, N, kk, gg, ee))
        print(f"{name:>22}  {ep_str:>5}  {N:>5}  {G[0]:8.3f}±{dG[0]:6.3f}")

    out = Path(f"results/L{L_sample}_crossL_3d_k{k}_propagator.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "epoch", "N_samples", "k_lat", "G", "dG"])
        for row in rows:
            w.writerow([row[0], row[1], row[2],
                        f"{row[3]:.10g}", f"{row[4]:.10g}", f"{row[5]:.10g}"])
    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
