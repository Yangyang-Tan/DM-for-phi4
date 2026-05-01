"""
High-precision (4096-sample) diagonal propagator comparison
between ep=8554 and ep=10000 multi-L cross-L samples vs HMC L=128.

Two figures:
  (a) log-log G(|k̂|), with per-bin ratio panel underneath
  (b) log-x linear-y G(|k̂|)

Diagonal modes (n, n) for n = 1..L/2.  x-axis is |k̂| (lattice momentum), NOT k̂².
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py

import sys; sys.path.append(str(Path(__file__).resolve().parent))
from analyze_diagonal_propagator import (
    diagonal_propagator, fit_xi, load_dm, load_hmc,
)


def main():
    L = 128
    k = "0.2705"
    run = Path(f"runs/phi4_Lmulti8-16-32-64_k{k}_l0.022_ncsnpp")
    ref = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2"))[:8192]

    sources = []
    sources.append(("HMC L=128 (N=8192)", "C0", "o", "-", ref))
    for ep, color, marker in [("8554", "C2", "s"), ("10000", "C3", "^")]:
        f = run / "data_crossL" / f"samples_multiL64_no_lcond_sample128_em_linear_steps2000_ep{ep}_4096.npy"
        if not f.exists():
            print(f"[WARN] missing {f}")
            continue
        d = load_dm(f)
        sources.append((f"DM ep={ep} (N={d.shape[0]})", color, marker, "-", d))

    # Compute G(k) for each
    spec = {}
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator(d)
        Z, xi = fit_xi(k_lat ** 2, G, n_low=5)
        spec[name] = (color, marker, ls, k_lat, G, E, xi)

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)

    # (a) log-log G + ratio panel
    ax = fig.add_subplot(gs[0, 0])
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.5, ms=5,
                    capsize=2,
                    label=f"{name}  ξ={xi:.1f}" if not np.isnan(xi) else name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title("log-log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
    ax.tick_params(labelbottom=False)

    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)
    name_hmc = next(iter(spec))
    G_hmc = spec[name_hmc][4]
    E_hmc = spec[name_hmc][5]
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        if name == name_hmc:
            ax_r.fill_between(k_lat, 1 - E_hmc / G_hmc, 1 + E_hmc / G_hmc,
                              color="C0", alpha=0.2, label="HMC ±1σ")
            ax_r.axhline(1.0, color="C0", lw=1.0)
            continue
        ratio = G / G_hmc
        # combined 1σ band (DM err + HMC err in quadrature, normalised)
        rerr = ratio * np.sqrt((E / G) ** 2 + (E_hmc / G_hmc) ** 2)
        ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                      lw=1.4, ms=4, capsize=2, label=name.split(" (")[0])
    ax_r.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax_r.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax_r.set_ylim(0.78, 1.15)
    ax_r.grid(alpha=0.3, which="both")
    ax_r.legend(fontsize=8, loc="lower left")

    # (b) log-x linear-y G
    ax2 = fig.add_subplot(gs[:, 1])
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        ax2.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.5, ms=5,
                     capsize=2, label=name)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax2.set_ylabel(r"$G(|\hat k|)$")
    ax2.set_title("log-x, linear-y")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "L=128 diagonal propagator (high-precision, 4096 cross-L samples each)\n"
        "linear schedule, 2000 SDE steps, multi-L L∈[8,16,32,64] σ=360 ckpt",
        y=1.02, fontsize=12)
    out = Path("results/crossL/diagonal_compare_4096.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
