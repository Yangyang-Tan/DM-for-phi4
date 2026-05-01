"""
In-distribution comparison at L=64: multi-L L=[8,16,32,64] vs L64-only,
both at ep=8554, against HMC L=64 ground truth.

Metrics (panels):
  (1) Diagonal G(|k̂|), log-log + ratio sub-panel
  (2) Magnetisation M = ⟨φ⟩_cfg distribution
  (3) Action S = Σ [-2κ φ Σ_nb φ + (1-2λ) φ² + λ φ⁴] distribution
  (4) Single-site φ histogram

Sigmas differ (multi-L σ=360 vs L64-only σ=250) — read result as
"which checkpoint better reproduces L=64 statistics", not as a controlled
training-data ablation.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_diagonal_propagator import (
    diagonal_propagator, fit_xi, load_dm, load_hmc,
)


def magnetisation(cfgs: np.ndarray) -> np.ndarray:
    """⟨φ⟩ per config — signed."""
    return cfgs.mean(axis=(1, 2))


def action(cfgs: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    """2D phi^4 lattice action per config (physical units)."""
    nb = np.roll(cfgs, 1, axis=1) + np.roll(cfgs, 1, axis=2)
    return (-2.0 * kappa * cfgs * nb
            + (1.0 - 2.0 * lam) * cfgs**2
            + lam * cfgs**4).sum(axis=(1, 2))


def hist_with_err(x: np.ndarray, bins: np.ndarray, n_boot: int = 200, seed: int = 0):
    """Histogram density with bootstrap 1σ band."""
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(bins) - 1))
    N = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        h, _ = np.histogram(x[idx], bins=bins, density=True)
        boots[b] = h
    return boots.mean(axis=0), boots.std(axis=0)


def main():
    L = 64
    kappa = 0.2705
    lam = 0.022
    k_str = f"{kappa}"

    multiL_dir = Path(f"runs/phi4_Lmulti8-16-32-64_k{k_str}_l0.022_ncsnpp")
    L64_dir    = Path(f"runs/phi4_L64_k{k_str}_l0.022_ncsnpp_sigma250")

    f_multi = multiL_dir / "data_crossL" / "samples_inDistL64_multiL_ep8554_em_log_steps2000_2048.npy"
    f_l64   = L64_dir    / "data_crossL" / "samples_inDistL64_L64only_sigma250_ep8554_em_log_steps2000_2048.npy"
    f_hmc   = Path(f"trainingdata/cfgs_wolff_fahmc_k={k_str}_l=0.022_{L}^2.jld2")

    for f in (f_multi, f_l64, f_hmc):
        if not f.exists():
            raise FileNotFoundError(f)

    hmc = load_hmc(f_hmc)
    if hmc.shape[0] > 8192:
        hmc = hmc[:8192]
    multi = load_dm(f_multi)
    l64   = load_dm(f_l64)

    sources = [
        ("HMC L=64",                 "C0", "o", "-", hmc),
        ("multi-L L=[8,16,32,64] σ=360 ep=8554", "C2", "s", "-", multi),
        ("L=64-only σ=250 ep=8554",  "C3", "v", "--", l64),
    ]

    # ---- (1) Diagonal propagator ----
    spec = {}
    print(f"\n{'method':>42}  {'N':>5}  {'ξ_fit':>7}  {'G(k_min)':>16}")
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator(d)
        _, xi = fit_xi(k_lat ** 2, G, n_low=5)
        spec[name] = (color, marker, ls, k_lat, G, E, xi)
        print(f"{name:>42}  {d.shape[0]:>5}  {xi:7.2f}  {G[0]:8.3f}±{E[0]:6.3f}")

    # ---- (2) Magnetisation, (3) Action, (4) phi histogram ----
    obs = {}
    for name, color, marker, ls, d in sources:
        M = magnetisation(d)
        S = action(d, kappa, lam) / d.shape[1] / d.shape[2]   # per-site for readability
        phi = d.reshape(-1)
        obs[name] = (color, marker, ls, M, S, phi)
        print(f"  {name:>42}: ⟨|M|⟩={np.mean(np.abs(M)):.4f}  "
              f"⟨S/V⟩={S.mean():.4f}  σ_S/V={S.std():.4f}")

    # ---------- Plot ----------
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 0.8, 2.0],
                          hspace=0.32, wspace=0.22)

    # G(k) log-log + ratio
    ax = fig.add_subplot(gs[0, 0])
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.5, ms=5,
                    capsize=2,
                    label=f"{name}  ξ={xi:.1f}" if not np.isnan(xi) else name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title(f"L={L}, κ={kappa}, λ={lam}  —  diagonal propagator")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(labelbottom=False)

    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)
    name_hmc = next(iter(spec))
    color_hmc, _, _, k_hmc, G_hmc, E_hmc, _ = spec[name_hmc]
    ax_r.fill_between(k_hmc, 1 - E_hmc / G_hmc, 1 + E_hmc / G_hmc,
                      color=color_hmc, alpha=0.2, label="HMC ±1σ")
    ax_r.axhline(1.0, color=color_hmc, lw=1.0)
    for name, (color, marker, ls, k_lat, G, E, xi) in spec.items():
        if name == name_hmc:
            continue
        ratio = G / G_hmc
        rerr = ratio * np.sqrt((E / G)**2 + (E_hmc / G_hmc)**2)
        ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                      lw=1.4, ms=4, capsize=2, label=name.split(" σ")[0])
    ax_r.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax_r.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax_r.set_ylim(0.7, 1.3)
    ax_r.grid(alpha=0.3, which="both")
    ax_r.legend(fontsize=8, loc="lower left")

    # Magnetisation
    ax2 = fig.add_subplot(gs[0, 1])
    M_all = np.concatenate([obs[n][3] for n in obs])
    bins_M = np.linspace(M_all.min(), M_all.max(), 51)
    bc_M = 0.5 * (bins_M[1:] + bins_M[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(M, bins_M)
        ax2.plot(bc_M, h, ls, color=color, lw=1.6,
                 label=f"{name.split(' σ')[0]}  ⟨|M|⟩={np.mean(np.abs(M)):.3f}")
        ax2.fill_between(bc_M, h - herr, h + herr, color=color, alpha=0.2)
    ax2.set_xlabel(r"$M = \langle\phi\rangle_{\rm cfg}$")
    ax2.set_ylabel("density")
    ax2.set_title("Magnetisation distribution")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    # Action per site
    ax3 = fig.add_subplot(gs[2, 0])
    S_all = np.concatenate([obs[n][4] for n in obs])
    bins_S = np.linspace(np.percentile(S_all, 0.5), np.percentile(S_all, 99.5), 60)
    bc_S = 0.5 * (bins_S[1:] + bins_S[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(S, bins_S)
        ax3.plot(bc_S, h, ls, color=color, lw=1.6,
                 label=f"{name.split(' σ')[0]}  ⟨S/V⟩={S.mean():.3f}")
        ax3.fill_between(bc_S, h - herr, h + herr, color=color, alpha=0.2)
    ax3.set_xlabel(r"$S / V$  (action per site)")
    ax3.set_ylabel("density")
    ax3.set_title("Action-per-site distribution")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    # Single-site phi
    ax4 = fig.add_subplot(gs[2, 1])
    phi_all = np.concatenate([obs[n][5][:200_000] for n in obs])
    bins_phi = np.linspace(np.percentile(phi_all, 0.1),
                           np.percentile(phi_all, 99.9), 80)
    bc_phi = 0.5 * (bins_phi[1:] + bins_phi[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        # subsample for speed
        if len(phi) > 500_000:
            phi = phi[np.random.default_rng(0).choice(len(phi), 500_000, replace=False)]
        h, herr = hist_with_err(phi, bins_phi)
        ax4.plot(bc_phi, h, ls, color=color, lw=1.6, label=name.split(" σ")[0])
        ax4.fill_between(bc_phi, h - herr, h + herr, color=color, alpha=0.2)
    ax4.set_xlabel(r"$\phi$ (single-site)")
    ax4.set_ylabel("density")
    ax4.set_title("Single-site $\\phi$ distribution")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"In-distribution L={L} comparison: multi-L {{8,16,32,64}}  vs  L=64-only  (ep=8554)",
        y=0.995, fontsize=13)

    out = Path(f"results/crossL/L{L}_indist_compare_ep8554.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
