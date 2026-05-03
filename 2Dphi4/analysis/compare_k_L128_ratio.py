"""
Side-by-side ratio-panel comparison of κ=0.2705 vs κ=0.27088 multi-L → L=128.

Purpose: dispel the visual illusion that κ=0.2705 result is closer to HMC.
The two existing plots used different axis layouts (the κ=0.2705 plot had no
ratio panel, hiding the IR deviation). Here we use IDENTICAL axes for both.

Both panels use:
- multi-L L=[8,16,32,64] σ=360 / σ=250 ckpt ep=10000
- LOG schedule, 2000 EM steps (controlled)
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_diagonal_propagator import diagonal_propagator, fit_xi, load_dm, load_hmc


def plot_one(ax_top, ax_ratio, hmc, dm, kappa, sigma_train, color_dm):
    k_lat, G_h, E_h = diagonal_propagator(hmc)
    k_lat_d, G_d, E_d = diagonal_propagator(dm)
    _, xi_h = fit_xi(k_lat ** 2, G_h, n_low=5)
    _, xi_d = fit_xi(k_lat_d ** 2, G_d, n_low=5)

    ax_top.errorbar(k_lat, G_h, yerr=E_h, fmt="o-", color="C0", lw=1.5, ms=5,
                    capsize=2, label=f"HMC L=128  ξ={xi_h:.1f}  N={hmc.shape[0]}")
    ax_top.errorbar(k_lat_d, G_d, yerr=E_d, fmt="s-", color=color_dm, lw=1.5, ms=5,
                    capsize=2,
                    label=f"multi-L σ={sigma_train} → L=128  ξ={xi_d:.1f}  N={dm.shape[0]}")
    ax_top.set_xscale("log"); ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax_top.set_title(f"κ={kappa}")
    ax_top.grid(alpha=0.3, which="both")
    ax_top.legend(fontsize=9, loc="lower left")
    ax_top.tick_params(labelbottom=False)

    ratio = G_d / G_h
    rerr = ratio * np.sqrt((E_d / G_d) ** 2 + (E_h / G_h) ** 2)
    ax_ratio.fill_between(k_lat, 1 - E_h / G_h, 1 + E_h / G_h,
                          color="C0", alpha=0.2)
    ax_ratio.axhline(1.0, color="C0", lw=1.0)
    ax_ratio.errorbar(k_lat_d, ratio, yerr=rerr, fmt="s-", color=color_dm,
                      lw=1.4, ms=4, capsize=2)
    ax_ratio.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax_ratio.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax_ratio.set_ylim(0.6, 1.5)
    ax_ratio.grid(alpha=0.3, which="both")
    ax_ratio.text(0.02, 0.05, f"G(k_min)/HMC = {ratio[0]:.3f}",
                  transform=ax_ratio.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))


def main():
    # κ=0.2705 (LOG schedule, ep=10000) — pick the file we already have
    hmc_2705 = load_hmc(Path("trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2"))[:8192]
    dm_2705 = load_dm(Path("runs/phi4_Lmulti8-16-32-64_k0.2705_l0.022_ncsnpp/data_crossL/"
                           "samples_multiL64_no_lcond_sample128_em_log_steps2000_ep10000.npy"))

    # κ=0.27088
    hmc_27088 = load_hmc(Path("trainingdata/cfgs_wolff_fahmc_k=0.27088_l=0.022_128^2.jld2"))[:8192]
    dm_27088 = load_dm(Path("runs/phi4_Lmulti8-16-32-64_k0.27088_l0.022_ncsnpp/data_crossL/"
                            "samples_crossL128_multiL_ep10000_em_log_steps2000_2048.npy"))

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.22)

    plot_one(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
             hmc_2705, dm_2705, kappa="0.2705", sigma_train=360, color_dm="C2")
    plot_one(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
             hmc_27088, dm_27088, kappa="0.27088", sigma_train=250, color_dm="C3")

    fig.suptitle("multi-L {8,16,32,64} → L=128 cross-L (OOD)  —  "
                 "log schedule, 2000 EM steps, ep=10000, identical axis style",
                 y=1.00, fontsize=12)

    out = Path("results/crossL/k_compare_L128_ratio.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
