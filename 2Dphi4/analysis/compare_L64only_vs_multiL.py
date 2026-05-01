"""
Compare cross-L L=128 propagator: only-L=64 training vs L=8-64 multi-L mix.

Both sampled with linear schedule + 2000 SDE steps, 4096 configurations.
Diagonal modes (n_x = n_y), x-axis is |k̂| (lattice momentum), NOT k̂².

Sources
-------
  HMC L=128 reference                 : ground truth (8192 cfgs)
  L=64-only σ=30  ckpt ep=9499        : pre-existing single-L baseline
  L=64-only σ=360 ckpt (latest)       : fair-σ single-L baseline (training)
  L=8-64 multi-L σ=360 ep=8554        : the multi-L model
  L=8-64 multi-L σ=360 ep=10000       : the multi-L model (later epoch)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.append(str(Path(__file__).resolve().parent))
from analyze_diagonal_propagator import (
    diagonal_propagator, fit_xi, load_dm, load_hmc,
)


def find_latest_ckpt_ep(run_dir: Path):
    """Return integer epoch number of latest ckpt under run_dir/models/."""
    eps = []
    for f in (run_dir / "models").glob("*.ckpt"):
        # match epoch=NNNN.ckpt or epoch=epoch=NNNN.ckpt
        for tok in f.stem.split("="):
            if tok.isdigit():
                eps.append(int(tok))
    return max(eps) if eps else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default="results/crossL/L64only_vs_multiL_4096.pdf")
    args = p.parse_args()

    L = 128
    k = "0.2705"
    multiL_dir   = Path(f"runs/phi4_Lmulti8-16-32-64_k{k}_l0.022_ncsnpp")
    L64s30_dir   = Path(f"runs/phi4_L64_k{k}_l0.022_ncsnpp")
    L64s360_dir  = Path(f"runs/phi4_L64_k{k}_l0.022_ncsnpp_sigma360")

    sources = [("HMC L=128 (N=8192)", "C0", "o", "-",
                load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2"))[:8192])]

    # L=64-only σ=30
    f = L64s30_dir / "data_crossL" / f"samples_L64only_sigma30_sample128_em_linear_steps2000_ep9499_4096.npy"
    if f.exists():
        sources.append((f"L=64-only σ=30 ep=9499 (4096)", "C1", "x", "--",
                        load_dm(f)))
    else:
        print(f"[WARN] missing {f}")

    # L=64-only σ=360
    if L64s360_dir.exists():
        ep = find_latest_ckpt_ep(L64s360_dir)
        if ep is not None:
            f = L64s360_dir / "data_crossL" / f"samples_L64only_sigma360_sample128_em_linear_steps2000_ep{ep}_4096.npy"
            if f.exists():
                sources.append((f"L=64-only σ=360 ep={ep} (4096)",
                                "C3", "v", "--", load_dm(f)))
            else:
                print(f"[INFO] L=64-only σ=360 ckpt available at ep={ep} but no samples yet ({f})")

    # Multi-L
    for ep, color, marker in [("8554", "C2", "s"), ("10000", "C5", "^")]:
        f = multiL_dir / "data_crossL" / f"samples_multiL64_no_lcond_sample128_em_linear_steps2000_ep{ep}_4096.npy"
        if f.exists():
            sources.append((f"multi-L L=[8,16,32,64] σ=360 ep={ep} (4096)",
                            color, marker, "-", load_dm(f)))

    # Compute G(k)
    spec = {}
    print(f"\n{'method':>50}  {'N':>5}  {'ξ_fit':>7}  {'G(k_min)':>16}  {'ratio':>14}")
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator(d)
        Z, xi = fit_xi(k_lat ** 2, G, n_low=5)
        spec[name] = (color, marker, ls, k_lat, G, E, xi)

    name_hmc = next(iter(spec))
    G_hmc = spec[name_hmc][4]
    E_hmc = spec[name_hmc][5]
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        ratio = G[0] / G_hmc[0]
        ratio_err = E[0] / G_hmc[0]
        print(f"{name:>50}  4096  {xi:7.2f}  {G[0]:8.3f}±{E[0]:6.3f}  {ratio:.4f}±{ratio_err:.4f}")

    # ---------- Plot ----------
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4,
                    ms=5, capsize=2,
                    label=f"{name}  ξ={xi:.1f}" if not np.isnan(xi) else name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  diagonal")
    ax.set_title("log-log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(labelbottom=False)

    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        if name == name_hmc:
            ax_r.fill_between(k_lat, 1 - E_hmc / G_hmc, 1 + E_hmc / G_hmc,
                              color="C0", alpha=0.2, label="HMC ±1σ")
            ax_r.axhline(1.0, color="C0", lw=1.0)
            continue
        ratio = G / G_hmc
        rerr = ratio * np.sqrt((E / G) ** 2 + (E_hmc / G_hmc) ** 2)
        ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                      lw=1.3, ms=4, capsize=2, label=name.split(" σ=")[0] + " " +
                      (name.split("σ=")[1].split(" ")[0] if "σ=" in name else ""))
    ax_r.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax_r.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax_r.set_ylim(0.55, 1.4)
    ax_r.grid(alpha=0.3, which="both")
    ax_r.legend(fontsize=8, loc="lower left")

    ax2 = fig.add_subplot(gs[:, 1])
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        ax2.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4,
                     ms=5, capsize=2, label=name)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax2.set_ylabel(r"$G(|\hat k|)$")
    ax2.set_title("log-x, linear-y")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "L=128 cross-L diagonal propagator:  only-L=64  vs  L=8-64 multi-L",
        y=1.02, fontsize=13)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
