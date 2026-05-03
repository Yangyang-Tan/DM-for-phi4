"""
Compare propagator scaling behaviour at L=128 between κ=0.2705 (closer to
finite-L pseudo-critical) and κ=0.27088 (closer to L→∞ critical).

Goal: pick the κ whose HMC propagator has the best Ornstein–Zernike scaling
form G(k) ≈ Z/(k²+m²) over the widest range of low momenta. That κ provides
the cleanest training signal for a diffusion model studying scaling behaviour.

Procedure:
  1. Load HMC L=128 cfgs for both κ
  2. Compute diagonal momentum-space propagator G(|k̂|) with bootstrap errors
  3. Fit OZ form Z/(k̂²+m²) using n_low = 3, 5, 8, 12, 16 lowest k modes
  4. Report fit χ²/dof, ξ=1/m, residuals
  5. Plot side-by-side
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_diagonal_propagator import diagonal_propagator, load_hmc


def fit_OZ(k_lat, G, E, n_low):
    """Fit G = Z/(k²+m²) using inverse-variance weighted least squares.

    Linearise: 1/G = (k²+m²)/Z = k²/Z + m²/Z. Linear in (1/Z, m²/Z).
    Returns (Z, m², χ²_red, dof).
    """
    k2 = k_lat[:n_low] ** 2
    G_sub = G[:n_low]
    E_sub = E[:n_low]
    y = 1.0 / G_sub
    # error on 1/G via propagation: σ(1/G) = σ(G) / G²
    sy = E_sub / G_sub ** 2
    w = 1.0 / sy ** 2
    # design matrix X = [k², 1], parameters (a, b) where a=1/Z, b=m²/Z
    X = np.column_stack([k2, np.ones_like(k2)])
    WX = X * w[:, None]
    A = WX.T @ X
    rhs = WX.T @ y
    a, b = np.linalg.solve(A, rhs)
    Z = 1.0 / a
    m2 = b * Z
    # χ²_red
    pred = X @ np.array([a, b])
    chi2 = ((y - pred) ** 2 * w).sum()
    dof = n_low - 2
    return Z, m2, chi2 / dof, dof


def main():
    L = 128
    cases = [
        ("κ=0.2705",  Path(f"trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_{L}^2.jld2"),  "C0", "o"),
        ("κ=0.27088", Path(f"trainingdata/cfgs_wolff_fahmc_k=0.27088_l=0.022_{L}^2.jld2"), "C3", "s"),
    ]

    # Load and compute propagator
    spec = {}
    for name, path, color, marker in cases:
        cfgs = load_hmc(path)
        if cfgs.shape[0] > 8192:
            cfgs = cfgs[:8192]
        k_lat, G, E = diagonal_propagator(cfgs)
        spec[name] = dict(color=color, marker=marker, k=k_lat, G=G, E=E,
                          N=cfgs.shape[0])
        print(f"\n{name}:  N={cfgs.shape[0]}  G(k_min)={G[0]:.3f}±{E[0]:.3f}  "
              f"k_min={k_lat[0]:.4f}  k_max={k_lat[-1]:.3f}  ({len(k_lat)} modes)")

    # Fit OZ at multiple n_low
    print(f"\n{'κ':>10}  {'n_low':>5}  {'ξ':>7}  {'Z':>9}  {'χ²/dof':>9}  "
          f"{'k_max_fit':>10}  {'k_max·ξ':>8}")
    print("-" * 75)
    fit_table = {}
    for name in spec:
        s = spec[name]
        fit_table[name] = []
        for n_low in [3, 5, 8, 12, 16]:
            Z, m2, chi2_red, dof = fit_OZ(s["k"], s["G"], s["E"], n_low)
            xi = 1.0 / np.sqrt(m2) if m2 > 0 else float('nan')
            k_max_fit = s["k"][n_low - 1]
            fit_table[name].append(dict(n_low=n_low, Z=Z, xi=xi, chi2_red=chi2_red,
                                         dof=dof, k_max_fit=k_max_fit))
            print(f"{name:>10}  {n_low:>5}  {xi:>7.2f}  {Z:>9.3f}  {chi2_red:>9.2f}  "
                  f"{k_max_fit:>10.4f}  {k_max_fit*xi:>8.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: G(k) log-log with OZ fit overlay (n_low=8)
    ax = axes[0]
    for name in spec:
        s = spec[name]
        ax.errorbar(s["k"], s["G"], yerr=s["E"], fmt=s["marker"]+"-",
                    color=s["color"], lw=1.4, ms=5, capsize=2,
                    label=f"{name}  HMC (N={s['N']})")
        # OZ fit overlay using n_low=8
        fit = next(f for f in fit_table[name] if f["n_low"] == 8)
        Z, xi = fit["Z"], fit["xi"]
        k_dense = np.geomspace(s["k"][0], s["k"][-1], 200)
        G_fit = Z / (k_dense**2 + 1.0/xi**2)
        ax.plot(k_dense, G_fit, "--", color=s["color"], lw=1.0, alpha=0.7,
                label=f"  fit ξ={xi:.2f}, Z={Z:.2f} (n_low=8)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
    ax.set_title(f"L={L} HMC propagator vs Ornstein–Zernike fit")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")

    # Right: residuals (G_fit / G_HMC) for OZ fit
    ax = axes[1]
    for name in spec:
        s = spec[name]
        fit = next(f for f in fit_table[name] if f["n_low"] == 8)
        Z, xi = fit["Z"], fit["xi"]
        G_fit = Z / (s["k"]**2 + 1.0/xi**2)
        ratio = G_fit / s["G"]
        rerr = ratio * s["E"] / s["G"]
        ax.errorbar(s["k"], ratio, yerr=rerr, fmt=s["marker"]+"-",
                    color=s["color"], lw=1.4, ms=5, capsize=2, label=f"{name}")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\hat k|$  (lattice units)")
    ax.set_ylabel(r"$G_{\rm OZ\,fit}(k) / G_{\rm HMC}(k)$")
    ax.set_title("OZ fit residual (n_low=8)")
    ax.set_ylim(0.7, 1.3)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)

    fig.suptitle(
        f"L={L}  κ scaling comparison:  κ=0.2705 (near pseudo-critical at finite L)  "
        f"vs  κ=0.27088 (near L→∞ critical)",
        y=1.00, fontsize=11)

    out = Path(f"results/kappa_scaling_L{L}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
