"""
3D L=64 cross-L propagator comparison for k ∈ {0.18, 0.2} multi-L runs.

Two DM cases:
    multi-L L=[4,8]    (sigma=100, ep=10000)
    multi-L L=[4,8,16] (sigma=280, ep=10000)
vs HMC L=64 reference at the same k.

EM/log/steps=2000, n=512 cfgs each.
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
    p.add_argument("--k", type=float, required=True, choices=[0.18, 0.2])
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--ep", type=int, default=10000)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="log",
                   choices=["log", "linear"])
    args = p.parse_args()

    L = 64
    suffix = f"em_{args.schedule}_steps{args.num_steps}_ep{args.ep}"
    f_hmc = Path(f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.lam}_{L}^3.jld2")
    f_4_8 = Path(
        f"runs/phi4_3d_Lmulti4-8_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
        f"samples_crossL_train8_sample{L}_{suffix}.npy"
    )
    f_4_8_16 = Path(
        f"runs/phi4_3d_Lmulti4-8-16_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
        f"samples_crossL_train16_sample{L}_{suffix}.npy"
    )
    f_4_8_16_32 = Path(
        f"runs/phi4_3d_Lmulti4-8-16-32_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
        f"samples_crossL_train32_sample{L}_{suffix}.npy"
    )
    for f in (f_hmc, f_4_8, f_4_8_16, f_4_8_16_32):
        if not f.exists():
            raise FileNotFoundError(f)

    cases = [
        ("HMC L=64",               "C0", "o", "-",  load_hmc_3d(f_hmc, max_n=4096)),
        ("multi-L L=[4,8]",        "C1", "s", "-",  load_dm_3d(f_4_8)),
        ("multi-L L=[4,8,16]",     "C2", "^", "-",  load_dm_3d(f_4_8_16)),
        ("multi-L L=[4,8,16,32]",  "C3", "D", "-",  load_dm_3d(f_4_8_16_32)),
    ]

    print(f"\nκ={args.k}, λ={args.lam}, L={L}, ep={args.ep}")
    print(f"{'method':>22} {'N':>5} {'G(k_min)':>16} {'⟨|M|⟩':>8} {'⟨S/V⟩':>9} {'σ_S/V':>7}")
    print("-" * 80)
    spec = {}
    obs = {}
    for name, color, marker, ls, d in cases:
        k_lat, G, E = diagonal_propagator_3d(d)
        M = magnetisation(d)
        S = action_3d(d, args.k, args.lam) / d.shape[1] ** 3
        spec[name] = (color, marker, ls, k_lat, G, E)
        obs[name]  = (color, marker, ls, M, S, d.reshape(-1))
        print(f"{name:>22} {d.shape[0]:>5} {G[0]:8.3f}±{E[0]:6.3f} "
              f"{np.mean(np.abs(M)):8.4f} {S.mean():9.4f} {S.std():7.4f}")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 0.8, 2.0],
                          hspace=0.32, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.5, ms=5,
                    capsize=2, label=f"{name}  G(k_min)={G[0]:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title(f"3D L={L} cross-L (OOD), κ={args.k}, λ={args.lam}, "
                 f"EM/{args.schedule}/steps={args.num_steps}, ep={args.ep}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
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
    ax_r.set_ylim(0.4, 1.6)
    ax_r.grid(alpha=0.3, which="both")

    ax2 = fig.add_subplot(gs[0, 1])
    M_all = np.concatenate([obs[n][3] for n in obs])
    bins_M = np.linspace(M_all.min(), M_all.max(), 51)
    bc_M = 0.5 * (bins_M[1:] + bins_M[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(M, bins_M)
        ax2.plot(bc_M, h, ls, color=color, lw=1.6,
                 label=f"{name}  ⟨|M|⟩={np.mean(np.abs(M)):.3f}")
        ax2.fill_between(bc_M, h - herr, h + herr, color=color, alpha=0.2)
    ax2.set_xlabel(r"$M = \langle\phi\rangle_{\rm cfg}$")
    ax2.set_ylabel("density"); ax2.set_title("Magnetisation")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2, 0])
    S_all = np.concatenate([obs[n][4] for n in obs])
    bins_S = np.linspace(np.percentile(S_all, 0.5), np.percentile(S_all, 99.5), 60)
    bc_S = 0.5 * (bins_S[1:] + bins_S[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(S, bins_S)
        ax3.plot(bc_S, h, ls, color=color, lw=1.6,
                 label=f"{name}  ⟨S/V⟩={S.mean():.3f}")
        ax3.fill_between(bc_S, h - herr, h + herr, color=color, alpha=0.2)
    ax3.set_xlabel(r"$S / V$"); ax3.set_ylabel("density")
    ax3.set_title("Action per site")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[2, 1])
    phi_all = np.concatenate([obs[n][5][:200_000] for n in obs])
    bins_phi = np.linspace(np.percentile(phi_all, 0.1),
                           np.percentile(phi_all, 99.9), 80)
    bc_phi = 0.5 * (bins_phi[1:] + bins_phi[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        if len(phi) > 500_000:
            phi = phi[np.random.default_rng(0).choice(len(phi), 500_000, replace=False)]
        h, herr = hist_with_err(phi, bins_phi)
        ax4.plot(bc_phi, h, ls, color=color, lw=1.6, label=name)
        ax4.fill_between(bc_phi, h - herr, h + herr, color=color, alpha=0.2)
    ax4.set_xlabel(r"$\phi$ (single-site)"); ax4.set_ylabel("density")
    ax4.set_title(r"Single-site $\phi$")
    ax4.grid(alpha=0.3); ax4.legend(fontsize=8)

    fig.suptitle(
        f"3D φ⁴ cross-L L={L} OOD: multi-L {{4,8}} vs {{4,8,16}}  vs  HMC  "
        f"(κ={args.k}, λ={args.lam}, ep={args.ep})",
        y=0.995, fontsize=13)

    out = Path(f"results/L{L}_crossL_3d_k{args.k}_ep{args.ep}_em_{args.schedule}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
