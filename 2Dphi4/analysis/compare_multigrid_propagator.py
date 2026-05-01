"""
Compare L=128 propagator across four sources:
  HMC reference                          (ground truth)
  no_lcond cross-L direct (no MG)        SDE only at L=128, OOD score
  bilinear-upsampled L=64                 multi-grid, no MALA
  bilinear-upsampled + MALA refined       multi-grid + MALA (action-gradient)

Also: per-bin G(k̂²_x)/G_HMC ratio, ξ_fit, S/V, std(φ).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn.functional as F


def load_dm(path: Path) -> np.ndarray:
    return np.load(path).transpose(2, 0, 1).astype(np.float64)


def load_hmc(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs


def per_config_pk(cfgs: np.ndarray) -> np.ndarray:
    L = cfgs.shape[1]
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    return (fk * fk.conj()).real / V


def axial_G(pk: np.ndarray, L: int):
    n_x = np.arange(1, L // 2 + 1)
    k2 = 4 * np.sin(np.pi * n_x / L) ** 2
    g_per_cfg = pk[:, n_x, 0]
    rng = np.random.default_rng(0)
    n_boot = 200
    N = pk.shape[0]
    boots = np.empty((n_boot, g_per_cfg.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k2, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k2, G, n_low=5):
    if len(k2) < n_low:
        return np.nan
    invG = 1.0 / G[:n_low]
    a, b = np.polyfit(k2[:n_low], invG, 1)
    if a <= 0 or b <= 0:
        return np.nan
    return 1.0 / np.sqrt(b / a)


def upsample_bilinear(cfgs_L64: np.ndarray, L_target: int) -> np.ndarray:
    """Bilinear upsample (N, L64, L64) → (N, L_target, L_target). Match the
    multigrid_mala.py upsampling exactly (same as torch.nn.functional)."""
    t = torch.from_numpy(cfgs_L64).float().unsqueeze(1)  # (N, 1, L, L)
    factor = L_target / cfgs_L64.shape[1]
    t = F.interpolate(t, scale_factor=factor, mode="bilinear",
                      align_corners=False)
    return t.squeeze(1).numpy().astype(np.float64)


def main():
    L = 128
    k = "0.2705"
    run_dir = Path(f"runs/phi4_Lmulti8-16-32-64_k{k}_l0.022_ncsnpp")
    ep = "6260"

    # Sources
    ref = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2"))
    if ref.shape[0] > 8192:
        ref = ref[:8192]

    cross_path = run_dir / "data_crossL" / f"samples_multiL64_no_lcond_sample{L}_em_log_steps2000_ep{ep}.npy"
    cross = load_dm(cross_path)

    mg_path = run_dir / "data_multigrid" / f"samples_multigrid_64to128_bilinear_h3e-04_mh1500_ep{ep}.npy"
    mg = load_dm(mg_path)

    # Recompute "upsampled, no MALA" from saved L=64 samples
    L64_path = run_dir / "data_crossL" / f"samples_multiL64_no_lcond_sample64_em_log_steps2000_ep{ep}.npy"
    L64 = load_dm(L64_path)
    if L64.shape[0] > mg.shape[0]:
        L64 = L64[:mg.shape[0]]
    up_only = upsample_bilinear(L64, L)

    sources = [
        ("HMC L=128",                "C0", "o", ref),
        ("cross-L direct (no MG)",   "C3", "s", cross),
        ("bilinear upsample only",   "gray", "x", up_only),
        ("multi-grid + MALA",        "C2", "^", mg),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rows = []
    print(f"{'method':>30}  {'N':>5}   {'ξ_fit':>7}  {'G(k_min)':>10}  {'ratio':>6}  {'<S/V>':>8}  {'std(φ)':>8}")
    for name, color, marker, d in sources:
        pk = per_config_pk(d)
        k2, G, E = axial_G(pk, L)
        xi = fit_xi(k2, G)
        # action density (compute on data; need k,l,phi_min,phi_max — but we can
        # compute the kernel directly on the loaded physical fields)
        kf, lf = float(k), 0.022
        nb = np.roll(d, 1, axis=1) + np.roll(d, 1, axis=2)
        s = (-2 * kf * d * nb + (1 - 2 * lf) * d**2 + lf * d**4)
        sv = s.sum(axis=(1, 2)) / (L * L)
        std_phi = d.std()
        ratio = G[0] / axial_G(per_config_pk(ref), L)[1][0]
        print(f"{name:>30}  {d.shape[0]:>5}   {xi:7.2f}  {G[0]:10.4f}  {ratio:6.3f}  "
              f"{sv.mean():.4f}±{sv.std():.4f}  {std_phi:.4f}")
        rows.append((name, color, marker, k2, G, E, xi, G[0]))

    # Plot: G vs k̂²
    ax = axes[0]
    G_hmc = rows[0][4]
    for name, color, marker, k2, G, E, xi, gmin in rows:
        ax.errorbar(k2, G, yerr=E, fmt=marker + "-", color=color, lw=1.4, ms=5,
                    label=f"{name}  (ξ_fit={xi:.1f})", capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\hat k_x^2$"); ax.set_ylabel(r"$G(\hat k_x^2)$")
    ax.set_title(f"L=128 axial propagator  (κ={k})")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.3, which="both")

    # Plot: ratio
    ax = axes[1]
    for name, color, marker, k2, G, E, xi, gmin in rows[1:]:  # skip HMC self
        ax.plot(k2, G / G_hmc, marker + "-", color=color, lw=1.4, ms=5, label=name)
    ax.axhline(1.0, color="C0", lw=1.0, alpha=0.7, label="HMC (ref)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\hat k_x^2$")
    ax.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    ax.set_ylim(0.0, 1.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Multi-grid + MALA refinement at L=128  "
        f"(L=64→L=128 bilinear, h=3e-4, 1500 MALA steps, acc≈89%)",
        y=1.02, fontsize=13)
    plt.tight_layout()
    out = Path("results/crossL/multigrid_compare_propagator.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
