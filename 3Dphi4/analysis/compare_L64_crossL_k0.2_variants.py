"""
3D L=64 cross-L propagator comparison for κ=0.2: all DM training variants vs HMC.

Variants:
    HMC L=64 (reference)
    multi-L L=[4,8]            (sigma=100, ep=10000)
    multi-L L=[4,8,16]         (sigma=280, ep=10000)
    multi-L L=[4,8,16,32]      (sigma=800, ep=10000, bs=128)
    multi-L L=[4,8,16,32]      (sigma=400, ep=10000, bs=128)
    multi-L L=[4,8,16,32]      (sigma=800, ep=20000, bs=256, resumed)

Variants with no sample file are skipped.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
    magnetisation, action_3d, hist_with_err,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="linear",
                   choices=["log", "linear"])
    args = p.parse_args()
    K = 0.2
    L = 64
    sx = f"em_{args.schedule}_steps{args.num_steps}"

    f_hmc = Path(f"trainingdata/cfgs_wolff_fahmc_k={K}_l={args.lam}_{L}^3.jld2")
    if not f_hmc.exists():
        raise FileNotFoundError(f_hmc)

    cases = [("HMC L=64", "C0", "o", "-", load_hmc_3d(f_hmc, max_n=4096))]

    paths = [
        ("L=[4,8]  σ=100",          "C1", "s", "-",
         f"runs/phi4_3d_Lmulti4-8_k{K}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train8_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16]  σ=280",       "C2", "^", "-",
         f"runs/phi4_3d_Lmulti4-8-16_k{K}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train16_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32]  σ=800 ep10k bs128",  "C3", "D", "-",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32]  σ=400 ep10k bs128",  "C4", "P", "-",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_sigma400/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32]  σ=200 ep10k bs128",  "C5", "*", "-",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_sigma200/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32]  σ=100 ep10k bs128",  "C6", "v", "-",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_sigma100/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep10000.npy"),
        ("L=[4,8,16,32]  σ=800 ep20k bs256 (resume)", "C7", "X", "-",
         f"runs/phi4_3d_Lmulti4-8-16-32_k{K}_l{args.lam}_ncsnpp_bs256/data_crossL/"
         f"samples_crossL_train32_sample{L}_{sx}_ep20000.npy"),
    ]
    for name, color, marker, ls, path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue
        cases.append((name, color, marker, ls, load_dm_3d(path)))

    spec, obs = {}, {}
    print(f"\nκ={K}, λ={args.lam}, L={L}, schedule={args.schedule}")
    print(f"{'method':>42} {'N':>5} {'G(k_min)':>16} {'⟨|M|⟩':>8} {'⟨S/V⟩':>9} {'σ_S/V':>7}")
    print("-" * 100)
    for name, color, marker, ls, d in cases:
        k_lat, G, E = diagonal_propagator_3d(d)
        M = magnetisation(d)
        S = action_3d(d, K, args.lam) / d.shape[1] ** 3
        spec[name] = (color, marker, ls, k_lat, G, E)
        obs[name] = (color, marker, ls, M, S, d.reshape(-1))
        print(f"{name:>42} {d.shape[0]:>5} {G[0]:8.3f}±{E[0]:6.3f} "
              f"{np.mean(np.abs(M)):8.4f} {S.mean():9.4f} {S.std():7.4f}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 0.9, 2.0],
                          hspace=0.32, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.5, ms=5,
                    capsize=2, label=f"{name}  G_min={G[0]:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title(f"3D L={L} cross-L (OOD), κ={K}, λ={args.lam}, "
                 f"EM/{args.schedule}/steps={args.num_steps}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.tick_params(labelbottom=False)

    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)
    name_hmc = next(iter(spec))
    color_hmc, _, _, k_hmc, G_hmc, E_hmc = spec[name_hmc]
    ax_r.fill_between(k_hmc, 1 - E_hmc / G_hmc, 1 + E_hmc / G_hmc,
                      color=color_hmc, alpha=0.2)
    ax_r.axhline(1.0, color=color_hmc, lw=1.0)
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        if name == name_hmc:
            continue
        ratio = G / G_hmc
        rerr = ratio * np.sqrt((E / G) ** 2 + (E_hmc / G_hmc) ** 2)
        ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                      lw=1.4, ms=4, capsize=2)
    ax_r.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax_r.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax_r.set_ylim(0.4, 5.0)
    ax_r.set_yscale("log")
    ax_r.grid(alpha=0.3, which="both")

    ax2 = fig.add_subplot(gs[0, 1])
    M_all = np.concatenate([obs[n][3] for n in obs])
    bins_M = np.linspace(M_all.min(), M_all.max(), 51)
    bc_M = 0.5 * (bins_M[1:] + bins_M[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(M, bins_M)
        ax2.plot(bc_M, h, ls, color=color, lw=1.4,
                 label=f"⟨|M|⟩={np.mean(np.abs(M)):.3f}")
        ax2.fill_between(bc_M, h - herr, h + herr, color=color, alpha=0.18)
    ax2.set_xlabel(r"$M = \langle\phi\rangle_{\rm cfg}$")
    ax2.set_ylabel("density"); ax2.set_title("Magnetisation")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[2, 0])
    S_all = np.concatenate([obs[n][4] for n in obs])
    bins_S = np.linspace(np.percentile(S_all, 0.5), np.percentile(S_all, 99.5), 60)
    bc_S = 0.5 * (bins_S[1:] + bins_S[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(S, bins_S)
        ax3.plot(bc_S, h, ls, color=color, lw=1.4,
                 label=f"⟨S/V⟩={S.mean():.3f}")
        ax3.fill_between(bc_S, h - herr, h + herr, color=color, alpha=0.18)
    ax3.set_xlabel(r"$S / V$"); ax3.set_ylabel("density")
    ax3.set_title("Action per site")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[2, 1])
    phi_all = np.concatenate([obs[n][5][:200_000] for n in obs])
    bins_phi = np.linspace(np.percentile(phi_all, 0.1),
                           np.percentile(phi_all, 99.9), 80)
    bc_phi = 0.5 * (bins_phi[1:] + bins_phi[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        if len(phi) > 500_000:
            phi = phi[np.random.default_rng(0).choice(len(phi), 500_000, replace=False)]
        h, herr = hist_with_err(phi, bins_phi)
        ax4.plot(bc_phi, h, ls, color=color, lw=1.4, label=name.split("  ")[0])
        ax4.fill_between(bc_phi, h - herr, h + herr, color=color, alpha=0.18)
    ax4.set_xlabel(r"$\phi$ (single-site)"); ax4.set_ylabel("density")
    ax4.set_title(r"Single-site $\phi$")
    ax4.grid(alpha=0.3); ax4.legend(fontsize=7)

    fig.suptitle(
        f"3D φ⁴ cross-L L={L} OOD κ={K}, λ={args.lam}: training-variant comparison "
        f"({args.schedule} schedule)",
        y=0.995, fontsize=13)

    out = Path(f"results/L{L}_crossL_3d_k{K}_variants_em_{args.schedule}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
