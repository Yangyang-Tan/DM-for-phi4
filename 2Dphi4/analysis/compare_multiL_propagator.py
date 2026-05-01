"""
Compare three score-models in momentum space, at three lattice sizes:
  - HMC reference                    (ground truth)
  - Single-L L=32 NCSNpp σ=180         (in-distribution at L=32, OOD at L=64,128)
  - Multi-L  L∈{8,16,32} NCSNpp σ=180  (multi-scale training)

For each L ∈ {32, 64, 128} we plot the axial G(k̂²_x) and the ratio DM/HMC.
The hypothesis: multi-L training should narrow the IR gap that single-L
showed at L=64 (DM/HMC ≈ 0.63 at the lowest mode), without hurting the
in-distribution L=32 quality.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py


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
    """1/G linear fit on lowest n_low bins → ξ = 1/√(b/a) where 1/G = a·k² + b/a."""
    if len(k2) < n_low:
        return np.nan
    invG = 1.0 / G[:n_low]
    a, b = np.polyfit(k2[:n_low], invG, 1)
    if a <= 0 or b <= 0:
        return np.nan
    return 1.0 / np.sqrt(b / a)


def main():
    Ls = [32, 64, 128]
    k = "0.2705"
    sigma_baseline_dir = Path(f"runs/phi4_L32_k{k}_l0.022_ncsnpp_sigma180")
    multiL_dir         = Path(f"runs/phi4_Lmulti8-16-32_k{k}_l0.022_ncsnpp")
    ep_baseline = "9111"
    ep_multiL   = "5000"

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"height_ratios": [2, 1]})
    rows = []

    for col, L in enumerate(Ls):
        # HMC reference
        ref = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2"))
        if ref.shape[0] > 8192:
            ref = ref[:8192]

        # Baseline single-L σ=180
        f_base = sigma_baseline_dir / "data_crossL" / f"samples_baseline_sigma180_sample{L}_em_log_steps2000_ep{ep_baseline}.npy"
        # Multi-L
        f_mul = multiL_dir / "data_crossL" / f"samples_multiL_sample{L}_em_log_steps2000_ep{ep_multiL}.npy"

        runs = {"HMC": ref}
        if f_base.exists():
            runs["Single-L σ=180"] = load_dm(f_base)
        else:
            print(f"[WARN] missing baseline: {f_base}")
        if f_mul.exists():
            runs["Multi-L"] = load_dm(f_mul)
        else:
            print(f"[WARN] missing multi-L: {f_mul}")

        ax = axes[0, col]
        ax_ratio = axes[1, col]
        styles = {"HMC": ("C0", "o"), "Single-L σ=180": ("C3", "s"), "Multi-L": ("C2", "^")}
        ref_k2, ref_G, ref_E = axial_G(per_config_pk(ref), L)
        for name, d in runs.items():
            color, marker = styles[name]
            k2, G, E = axial_G(per_config_pk(d), L)
            ax.errorbar(k2, G, yerr=E, fmt=marker + "-", color=color, lw=1.4, ms=5,
                        label=f"{name} (N={d.shape[0]})", capsize=2)
            xi = fit_xi(k2, G)
            print(f"L={L:3d}  {name:<18}  ξ_fit={xi:6.2f}  G(k_min)={G[0]:8.4f}")
            rows.append((L, name, xi, G[0], ref_G[0]))
            if name != "HMC":
                ratio = G / ref_G
                ax_ratio.plot(k2, ratio, marker + "-", color=color, lw=1.4, ms=4,
                              label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"L={L}  (κ={k})", fontsize=12)
        ax.set_xlabel(r"$\hat k_x^2$"); ax.set_ylabel(r"$G(\hat k_x^2)$")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

        ax_ratio.axhline(1.0, color="C0", lw=1.0, alpha=0.7)
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel(r"$\hat k_x^2$")
        ax_ratio.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
        ax_ratio.set_ylim(0.5, 1.6)
        ax_ratio.grid(alpha=0.3, which="both")
        ax_ratio.legend(fontsize=8)

    fig.suptitle(
        f"Multi-L vs single-L cross-L extrapolation  (κ={k}, λ=0.022, σ=180)",
        y=1.00, fontsize=14)
    plt.tight_layout()
    out = Path("results/crossL/multiL_compare_propagator.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Summary table
    print(f"\n{'L':>3}  {'method':>20}   {'ξ_fit':>8}  {'G(k_min)':>10}  {'ratio':>6}")
    for r in rows:
        L, name, xi, gmin, gref = r
        ratio = gmin / gref if gref != 0 else np.nan
        print(f"{L:>3}  {name:>20}   {xi:8.2f}  {gmin:10.4f}  {ratio:6.3f}")


if __name__ == "__main__":
    main()
