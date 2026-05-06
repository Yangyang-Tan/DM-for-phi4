"""
Comprehensive 3D L=64 cross-L comparison: pair each L_train_max ∈ {8, 16, 32}
with its multi-L counterpart, all under linear-schedule SDE at ep=10000.

Lines on plot:
    HMC L=64 (reference)
    L=8-only            ↔  multi-L L=[4,8]
    L=16-only           ↔  multi-L L=[4,8,16]
    L=32-only           ↔  multi-L L=[4,8,16,32]

x-axis is dimensionless lattice momentum |k_lat|.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
    magnetisation, action_3d,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--ep", type=int, default=10000)
    args = p.parse_args()

    L_sample = 64
    crossL_tag = (f"crossL{L_sample}_{{name}}_ep{args.ep}_"
                  f"em_linear_steps2000_512")

    f_hmc = Path(f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.lam}_{L_sample}^3.jld2")
    cases = [
        ("HMC L=64",         "C0", "o", "-",  load_hmc_3d(f_hmc)),
    ]
    paths = [
        ("L=8-only",         "C1", "v", "--",
         f"runs/phi4_3d_L8_k{args.k}_l{args.lam}_ncsnpp_sigma100/data_crossL/"
         f"samples_crossL64_L8only_ep{args.ep}_em_linear_steps2000_512.npy"),
        ("multi-L L=[4,8]",  "C1", "s", "-",
         f"runs/phi4_3d_Lmulti4-8_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_multiL4-8_ep{args.ep}_em_linear_steps2000_512.npy"),
        ("L=16-only",        "C2", "v", "--",
         f"runs/phi4_3d_L16_k{args.k}_l{args.lam}_ncsnpp_sigma280/data_crossL/"
         f"samples_crossL64_L16only_ep{args.ep}_em_linear_steps2000_512.npy"),
        ("multi-L L=[4,8,16]","C2","s", "-",
         f"runs/phi4_3d_Lmulti4-8-16_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_multiL4-8-16_ep{args.ep}_em_linear_steps2000_512.npy"),
        ("L=32-only",        "C3", "v", "--",
         f"runs/phi4_3d_L32_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_L32only_ep{args.ep}_em_linear_steps2000_512.npy"),
        ("multi-L L=[4,8,16,32]","C3","s", "-",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL64_multiL_ep{args.ep}_em_linear_steps2000_512.npy"),
    ]
    for name, color, marker, ls, path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[skip] {name}: {path}")
            continue
        cases.append((name, color, marker, ls, load_dm_3d(path)))

    # Compute spectra and stats
    spec = {}
    print(f"\n{'method':>28}  {'N':>5}  {'G(k_min)':>16}  "
          f"{'⟨|M|⟩':>7}  {'⟨S/V⟩':>9}  {'σ_S/V':>7}")
    print("-" * 90)
    for name, color, marker, ls, d in cases:
        k_lat, G, E = diagonal_propagator_3d(d)
        M = magnetisation(d)
        S = action_3d(d, args.k, args.lam) / d.shape[1] ** 3
        spec[name] = (color, marker, ls, k_lat, G, E,
                      np.mean(np.abs(M)), S.mean(), S.std())
        print(f"{name:>28}  {d.shape[0]:>5}  {G[0]:8.3f}±{E[0]:6.3f}  "
              f"{np.mean(np.abs(M)):7.4f}  {S.mean():9.4f}  {S.std():7.4f}")

    # Plot: top G(k), bottom ratio panel
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    name_hmc = next(iter(spec))
    color_hmc, _, _, k_hmc, G_hmc, E_hmc, *_ = spec[name_hmc]
    for name, (color, marker, ls, k_lat, G, E, _, _, _) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4, ms=5,
                    capsize=2, label=f"{name}  G(k_min)={G[0]:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title(f"3D L={L_sample} cross-L (linear SDE, ep={args.ep}), "
                 f"κ={args.k}, λ={args.lam}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(labelbottom=False)

    ax_r.fill_between(k_hmc, 1 - E_hmc / G_hmc, 1 + E_hmc / G_hmc,
                      color=color_hmc, alpha=0.2, label="HMC ±1σ")
    ax_r.axhline(1.0, color=color_hmc, lw=1.0)
    for name, (color, marker, ls, k_lat, G, E, _, _, _) in spec.items():
        if name == name_hmc:
            continue
        ratio = G / G_hmc
        rerr = ratio * np.sqrt((E / G) ** 2 + (E_hmc / G_hmc) ** 2)
        ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                      lw=1.2, ms=4, capsize=2)
    ax_r.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax_r.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax_r.set_ylim(0.5, 1.5)
    ax_r.grid(alpha=0.3, which="both")

    out = Path(f"results/L{L_sample}_crossL_3d_all_k{args.k}_ep{args.ep}_em_linear.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
