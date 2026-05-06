"""
3D L=64 cross-L OOD propagator comparison: multi-L L=[4,8,16,32] vs L=32-only,
both at ep=10000, vs HMC L=64 reference. κ=0.1923, λ=0.9.

L=64 is OOD for both models (multi-L max training L=32, L=32-only training L=32).
Minimum lattice momentum 2π/64 ≈ 0.098 < 2π/32 = 0.196 (training min).

Diagonal modes: (n, n, n) for n = 1..L/2.
  k̂² = 3 · 4·sin²(π·n/L)
  |k̂| = sqrt(3) · 2·sin(π·n/L)
  G(k) = ⟨ |φ̃(k)|² ⟩ / V    with φ̃ = FFT3(φ - ⟨φ⟩)
  Bootstrap errors over cfgs (n_boot=200).
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dm_3d(path):
    """Load (L, L, L, N) → (N, L, L, L)."""
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[-1] > arr.shape[0]:
        arr = arr.transpose(3, 0, 1, 2)
    return arr.astype(np.float64)


def load_hmc_3d(path, max_n=4096):
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    if cfgs.ndim == 4 and cfgs.shape[-1] > cfgs.shape[0]:
        cfgs = cfgs.transpose(3, 0, 1, 2)
    return cfgs[:max_n]


def diagonal_propagator_3d(cfgs, n_boot=200, seed=0):
    """Diagonal G(|k̂|) for cfgs of shape (N, L, L, L)."""
    L = cfgs.shape[1]
    phi = cfgs - cfgs.mean()
    fk = np.fft.fftn(phi, axes=(1, 2, 3))
    g = (np.abs(fk) ** 2) / (L ** 3)   # (N, L, L, L)

    diag_n = np.arange(1, L // 2 + 1)
    g_diag = g[:, diag_n, diag_n, diag_n]    # (N, L/2)
    # k̂ on diagonal: |k̂| = sqrt(3) * 2*sin(πn/L)
    k_lat = np.sqrt(3.0) * 2.0 * np.sin(np.pi * diag_n / L)

    # Bootstrap
    rng = np.random.default_rng(seed)
    N = g_diag.shape[0]
    boots = np.empty((n_boot, len(diag_n)))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_diag[idx].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi_3d(k2, G, n_low=5):
    """OZ fit G = Z/(k²+m²) on n_low lowest k. Returns (Z, ξ=1/m)."""
    k2 = k2[:n_low]
    G_sub = G[:n_low]
    y = 1.0 / G_sub
    X = np.column_stack([k2, np.ones_like(k2)])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    Z = 1.0 / a
    m2 = b * Z
    if m2 <= 0:
        return float('nan'), float('nan')
    return Z, 1.0 / np.sqrt(m2)


def magnetisation(cfgs):
    return cfgs.mean(axis=(1, 2, 3))


def action_3d(cfgs, kappa, lam):
    """3D phi^4 action per cfg (sum over volume)."""
    nb = (np.roll(cfgs, 1, axis=1) + np.roll(cfgs, 1, axis=2) + np.roll(cfgs, 1, axis=3))
    return (-2.0 * kappa * cfgs * nb
            + (1.0 - 2.0 * lam) * cfgs ** 2
            + lam * cfgs ** 4).sum(axis=(1, 2, 3))


def hist_with_err(x, bins, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(bins) - 1))
    N = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        h, _ = np.histogram(x[idx], bins=bins, density=True)
        boots[b] = h
    return boots.mean(axis=0), boots.std(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--ep", type=int, default=10000)
    p.add_argument("--method", type=str, default="em", choices=["em", "ode"])
    p.add_argument("--schedule", type=str, default="log")
    p.add_argument("--num_steps", type=int, default=2000,
                   help="EM uses 2000, ODE typically 400")
    p.add_argument("--z2_symmetrize", action="store_true",
                   help="cat([s, -s]) on DM samples to enforce Z_2 exactness")
    args = p.parse_args()

    L = 64
    multiL_dir = Path(f"runs/phi4_3d_Lmulti4-8-16-32_k{args.k}_l{args.lam}_ncsnpp")
    L32_dir    = Path(f"runs/phi4_3d_L32_k{args.k}_l{args.lam}_ncsnpp")

    suffix = f"{args.method}_{args.schedule}_steps{args.num_steps}_512"
    f_multi = (multiL_dir / "data_crossL"
               / f"samples_crossL64_multiL_ep{args.ep}_{suffix}.npy")
    f_l32   = (L32_dir    / "data_crossL"
               / f"samples_crossL64_L32only_ep{args.ep}_{suffix}.npy")
    f_hmc   = Path(f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.lam}_{L}^3.jld2")

    for f in (f_multi, f_l32, f_hmc):
        if not f.exists():
            raise FileNotFoundError(f)

    hmc = load_hmc_3d(f_hmc, max_n=4096)
    multi = load_dm_3d(f_multi)
    l32 = load_dm_3d(f_l32)
    if args.z2_symmetrize:
        multi = np.concatenate([multi, -multi], axis=0)
        l32   = np.concatenate([l32,   -l32],   axis=0)
        hmc   = np.concatenate([hmc,   -hmc],   axis=0)
        print(f"[Z_2 symmetrized] multi N={multi.shape[0]}  l32 N={l32.shape[0]}  hmc N={hmc.shape[0]}\n")

    sources = [
        ("HMC L=64",                                   "C0", "o", "-",  hmc),
        (f"multi-L L=[4,8,16,32] → L=64 ep={args.ep}", "C2", "s", "-",  multi),
        (f"L=32-only → L=64 ep={args.ep}",             "C3", "v", "--", l32),
    ]

    spec = {}
    print(f"{'method':>50}  {'N':>5}  {'G(k_min)':>16}")
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator_3d(d)
        spec[name] = (color, marker, ls, k_lat, G, E)
        print(f"{name:>50}  {d.shape[0]:>5}  {G[0]:8.3f}±{E[0]:6.3f}")

    obs = {}
    print()
    for name, color, marker, ls, d in sources:
        M = magnetisation(d)
        S = action_3d(d, args.k, args.lam) / d.shape[1] ** 3
        phi = d.reshape(-1)
        obs[name] = (color, marker, ls, M, S, phi)
        print(f"{name:>50}: ⟨|M|⟩={np.mean(np.abs(M)):.4f}  "
              f"⟨S/V⟩={S.mean():.4f}  σ_S/V={S.std():.4f}")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 0.8, 2.0],
                          hspace=0.32, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.5, ms=5,
                    capsize=2, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title(f"3D L={L} cross-L (OOD), κ={args.k}, λ={args.lam}  —  diagonal propagator")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(labelbottom=False)

    ax_r = fig.add_subplot(gs[1, 0], sharex=ax)
    name_hmc = next(iter(spec))
    color_hmc, _, _, k_hmc, G_hmc, E_hmc = spec[name_hmc]
    ax_r.fill_between(k_hmc, 1 - E_hmc / G_hmc, 1 + E_hmc / G_hmc, color="C0", alpha=0.2)
    ax_r.axhline(1.0, color="C0", lw=1.0)
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
                 label=f"{name.split(' ep')[0]}  ⟨|M|⟩={np.mean(np.abs(M)):.3f}")
        ax2.fill_between(bc_M, h - herr, h + herr, color=color, alpha=0.2)
    ax2.set_xlabel(r"$M = \langle\phi\rangle_{\rm cfg}$")
    ax2.set_ylabel("density")
    ax2.set_title("Magnetisation distribution")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2, 0])
    S_all = np.concatenate([obs[n][4] for n in obs])
    bins_S = np.linspace(np.percentile(S_all, 0.5), np.percentile(S_all, 99.5), 60)
    bc_S = 0.5 * (bins_S[1:] + bins_S[:-1])
    for name, (color, marker, ls, M, S, phi) in obs.items():
        h, herr = hist_with_err(S, bins_S)
        ax3.plot(bc_S, h, ls, color=color, lw=1.6,
                 label=f"{name.split(' ep')[0]}  ⟨S/V⟩={S.mean():.3f}")
        ax3.fill_between(bc_S, h - herr, h + herr, color=color, alpha=0.2)
    ax3.set_xlabel(r"$S / V$  (action per site)")
    ax3.set_ylabel("density")
    ax3.set_title("Action-per-site distribution")
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
        ax4.plot(bc_phi, h, ls, color=color, lw=1.6, label=name.split(" ep")[0])
        ax4.fill_between(bc_phi, h - herr, h + herr, color=color, alpha=0.2)
    ax4.set_xlabel(r"$\phi$ (single-site)")
    ax4.set_ylabel("density")
    ax4.set_title(r"Single-site $\phi$ distribution")
    ax4.grid(alpha=0.3); ax4.legend(fontsize=8)

    z2_tag = "_z2sym" if args.z2_symmetrize else ""
    fig.suptitle(
        f"3D phi^4 cross-L L={L} OOD: multi-L {{4,8,16,32}}  vs  L=32-only  "
        f"(κ={args.k}, λ={args.lam}, ep={args.ep}, "
        f"{args.method.upper()}/{args.schedule}/steps={args.num_steps})",
        y=0.995, fontsize=13)

    out = Path(f"results/L{L}_crossL_compare_3d_k{args.k}_ep{args.ep}_"
               f"{args.method}_{args.schedule}_s{args.num_steps}{z2_tag}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
