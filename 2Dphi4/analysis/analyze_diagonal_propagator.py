"""
Diagonal momentum-space propagator G(k) for cross-L L=128 samples.

Conventions (match CorrelationUtils.jl in this repo)
----------------------------------------------------
  Lattice momentum:    k̂² = Σ_i 4 sin²(π n_i / L)
  Diagonal modes:      (n, n) for n = 1, ..., L/2  → diagonality = 0.5
  Propagator:          G(k) = ⟨ |φ̃(k)|² ⟩ / V    with φ̃ = FFT(φ - ⟨φ⟩)
  Bootstrap errors:    n_boot=200 over configurations

x-axis is |k̂| = √(k̂²), NOT k̂² (per user request).
Two figures:
  (a) log-log
  (b) log-x, linear-y
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse


def load_dm(path: Path) -> np.ndarray:
    return np.load(path).transpose(2, 0, 1).astype(np.float64)


def load_hmc(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs


def diagonal_propagator(cfgs: np.ndarray, n_boot: int = 200, seed: int = 0):
    """G(k) on diagonal lattice modes (n,n). Returns (k_lat, G_mean, G_err).

    k_lat = √(k̂²) where k̂² = 8 sin²(π n / L) on diagonal.
    """
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V                      # (N, L, L)

    n_diag = np.arange(1, L // 2 + 1)                  # skip n=0
    g_per_cfg = pk[:, n_diag, n_diag]                  # (N, L/2)

    k2 = 8.0 * np.sin(np.pi * n_diag / L) ** 2          # diagonal: 2 × 4sin²(πn/L)
    k_lat = np.sqrt(k2)

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, g_per_cfg.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k2, G, n_low: int = 5):
    """1/G = a k² + b → ξ = 1/√(b/a)."""
    if len(k2) < n_low:
        return float("nan"), float("nan")
    invG = 1.0 / G[:n_low]
    a, b = np.polyfit(k2[:n_low], invG, 1)
    if a <= 0 or b <= 0:
        return float("nan"), float("nan")
    Z = 1.0 / a
    m2 = b / a
    return Z, 1.0 / np.sqrt(m2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ep", type=str, default="8554",
                   help="ckpt epoch token used in sample filenames")
    p.add_argument("--out", type=str,
                   default="results/crossL/diagonal_compare_ep8554.pdf")
    args = p.parse_args()

    L = 128
    k_kappa = "0.2705"
    run_dir = Path(f"runs/phi4_Lmulti8-16-32-64_k{k_kappa}_l0.022_ncsnpp")

    sources = [("HMC L=128",                 "C0", "o", "-",
                load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k_kappa}_l=0.022_{L}^2.jld2"))[:8192])]

    for sched, color, marker in [("log",   "C3", "s"),
                                  ("linear","C2", "^")]:
        for steps, ls in [(2000, "-"), (4000, "--")]:
            tag = (f"multiL64_no_lcond_sample{L}_em_{sched}_steps{steps}_ep{args.ep}")
            f = run_dir / "data_crossL" / f"samples_{tag}.npy"
            if f.exists():
                d = load_dm(f)
                sources.append((f"DM {sched} steps={steps}",
                                color, marker, ls, d))
            else:
                print(f"[WARN] missing {f}")

    # Compute G(k) for each
    print(f"{'method':>30}  {'N':>5}  {'ξ_fit':>7}  {'G(k_min)':>14}")
    rows = []
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator(d)
        k2 = k_lat ** 2
        Z, xi = fit_xi(k2, G, n_low=5)
        print(f"{name:>30}  {d.shape[0]:>5}  {xi:7.2f}  {G[0]:8.4f}±{E[0]:6.4f}")
        rows.append((name, color, marker, ls, k_lat, G, E, xi))

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) log-log
    ax = axes[0]
    for name, color, marker, ls, k_lat, G, E, xi in rows:
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4,
                    ms=5, capsize=2,
                    label=f"{name}  (ξ={xi:.1f})" if not np.isnan(xi) else name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G(k)$  (diagonal modes, $n_x{=}n_y$)")
    ax.set_title(f"L={L},  κ={k_kappa},  λ=0.022,  ckpt ep={args.ep}\nlog-log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")

    # (b) log-x, linear-y
    ax = axes[1]
    for name, color, marker, ls, k_lat, G, E, xi in rows:
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color, lw=1.4,
                    ms=5, capsize=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G(k)$")
    ax.set_title("log-x, linear-y")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("Diagonal propagator (diagonality = 0.5, n_x = n_y)",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
