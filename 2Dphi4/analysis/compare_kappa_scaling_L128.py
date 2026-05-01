"""
Compare scaling behavior of two L=128 phi^4 training datasets:
  κ=0.27088 (claimed true critical point) vs κ=0.2705 (peak-χ pseudo-critical).

For each dataset we report:
  - Magnetic order: ⟨|m|⟩, χ = V·(⟨m²⟩ − ⟨|m|⟩²), Binder U_4 = 1 − ⟨m⁴⟩/(3⟨m²⟩²)
  - Radial momentum propagator G(k̂) with bootstrap error
  - Effective mass m_eff & correlation length ξ from low-k OPE fit
        1/G(k̂²) = (k̂² + m²)/Z   →   ξ = 1/m
  - Wall correlator G_w(t) (zero-momentum projection along x, transversal sum
    along y; per-cfg subtraction of global mean to kill the m≠0 zero mode)
  - Step effective mass m_step(t) = arccosh((G_w(t−1)+G_w(t+1))/(2 G_w(t)))

Outputs (to ./compare_kappa_scaling_L128/):
  - summary.txt   tabulated scalars + ξ
  - figure.png    G(k̂) log-log + G_w(t) semi-log + m_step(t)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "trainingdata"
OUT_DIR  = Path(__file__).resolve().parent / "compare_kappa_scaling_L128"
OUT_DIR.mkdir(exist_ok=True)

L      = 128
LAMBDA = 0.022
N_BOOT = 400
SEED   = 0

CASES = [
    ("0.27088", "k=0.27088 (true Tc)"),
    ("0.2705",  "k=0.2705 (peak-χ)"),
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_cfgs(kappa: str) -> np.ndarray:
    """Return cfgs of shape (N, L, L), float64."""
    path = DATA_DIR / f"cfgs_wolff_fahmc_k={kappa}_l={LAMBDA}_{L}^2.jld2"
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    if cfgs.shape[0] != max(cfgs.shape):
        cfgs = np.moveaxis(cfgs, int(np.argmax(cfgs.shape)), 0)
    assert cfgs.shape[1:] == (L, L), f"unexpected shape {cfgs.shape}"
    return cfgs


# ---------------------------------------------------------------------------
# Magnetic observables (per-config, then bootstrap over cfgs)
# ---------------------------------------------------------------------------
def magnetic_scalars(m_cfg: np.ndarray, V: int, n_boot: int = N_BOOT,
                     seed: int = SEED):
    """Return dict of (mean, err) for ⟨|m|⟩, χ, U_4."""
    rng = np.random.default_rng(seed)
    N   = m_cfg.size

    def stats(idx):
        m  = m_cfg[idx]
        am = np.mean(np.abs(m))
        m2 = np.mean(m**2)
        m4 = np.mean(m**4)
        chi = V * (m2 - am**2)
        u4  = 1.0 - m4 / (3.0 * m2**2)
        return am, chi, u4

    am0, chi0, u40 = stats(np.arange(N))
    boot = np.empty((n_boot, 3))
    for b in range(n_boot):
        boot[b] = stats(rng.integers(0, N, size=N))
    err = boot.std(axis=0)
    return {
        "abs_m":  (am0,  err[0]),
        "chi":    (chi0, err[1]),
        "U4":     (u40,  err[2]),
    }


# ---------------------------------------------------------------------------
# Radial momentum propagator with bootstrap
# ---------------------------------------------------------------------------
def lattice_kh2_grid(L: int):
    n  = np.arange(L)
    n  = np.where(n > L // 2, n - L, n)
    p  = 4.0 * np.sin(np.pi * n / L) ** 2
    KH2 = p[:, None] + p[None, :]
    sum_p4 = (p[:, None]) ** 2 + (p[None, :]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = np.where(KH2 > 1e-12, sum_p4 / KH2 ** 2, 0.0)
    return KH2, diag


def per_cfg_pk(cfgs: np.ndarray) -> np.ndarray:
    """G_i(k) = |φ̃_i(k)|² / V for each config i; subtract per-cfg mean."""
    L = cfgs.shape[-1]
    V = L * L
    phi = cfgs - cfgs.mean(axis=(1, 2), keepdims=True)
    fk  = np.fft.fft2(phi, axes=(1, 2))
    return (fk * fk.conj()).real / V


def radial_average_per_cfg(Gk_per_cfg: np.ndarray, KH2: np.ndarray,
                           mask: np.ndarray, n_bins: int):
    """Log-spaced |k̂| bins. Return (k_centres, per-cfg radial G of shape (N, nb))."""
    k_sel = KH2[mask]
    pos   = k_sel[k_sel > 1e-10]
    edges = np.geomspace(pos.min(), pos.max() * 1.001, n_bins + 1)
    centres_kh2 = np.sqrt(edges[:-1] * edges[1:])

    Gflat = Gk_per_cfg[:, mask]
    out   = np.full((Gk_per_cfg.shape[0], n_bins), np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        in_bin = (k_sel >= edges[b]) & (k_sel < edges[b + 1])
        c = in_bin.sum()
        if c == 0:
            continue
        counts[b] = c
        out[:, b] = Gflat[:, in_bin].mean(axis=1)
    valid = counts > 0
    return np.sqrt(centres_kh2[valid]), out[:, valid]


def bootstrap_mean_err(per_cfg: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    N   = per_cfg.shape[0]
    means = np.empty((n_boot, per_cfg.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        means[b] = per_cfg[idx].mean(axis=0)
    return per_cfg.mean(axis=0), means.std(axis=0)


def fit_xi(k_centres: np.ndarray, G_per_cfg: np.ndarray,
           n_low: int = 5, n_boot: int = N_BOOT, seed: int = SEED):
    """Fit  1/G(k̂²) = (k̂² + m²)/Z  on lowest n_low bins.  ξ = 1/m."""
    rng = np.random.default_rng(seed)
    N   = G_per_cfg.shape[0]
    k2  = (k_centres[:n_low]) ** 2

    def fit_once(Gmean_low):
        invG = 1.0 / Gmean_low
        a, b = np.polyfit(k2, invG, 1)
        if a <= 0 or b <= 0:
            return np.nan, np.nan, np.nan
        Z, m2 = 1.0 / a, b / a
        return Z, m2, 1.0 / np.sqrt(m2)

    Z0, m20, xi0 = fit_once(G_per_cfg.mean(axis=0)[:n_low])
    xi_b = np.full(n_boot, np.nan)
    m_b  = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        Z, m2, xi = fit_once(G_per_cfg[idx].mean(axis=0)[:n_low])
        xi_b[b] = xi
        m_b[b]  = np.sqrt(m2) if (m2 == m2 and m2 > 0) else np.nan
    return {
        "Z":   Z0,
        "m":   np.sqrt(m20),
        "m_err": np.nanstd(m_b),
        "xi":  xi0,
        "xi_err": np.nanstd(xi_b),
    }


# ---------------------------------------------------------------------------
# Wall correlator and step effective mass
# ---------------------------------------------------------------------------
def wall_correlator_per_cfg(cfgs: np.ndarray) -> np.ndarray:
    """
    G_w(t) = (1/L) Σ_x ⟨φ̃(x,0) φ̃(x,t)⟩ , φ̃(x,y)=φ(x,y) − ⟨φ⟩_cfg.
    Uses zero-mode projection along the y-axis: for each x,
        Φ_x = (1/L) Σ_y φ̃(x,y)  ;  G_w(t) = (1/L) Σ_x Φ_x · Φ_{x+t}
    Computed per cfg via FFT along x. Returns shape (N, L).
    """
    L  = cfgs.shape[-1]
    phi = cfgs - cfgs.mean(axis=(1, 2), keepdims=True)
    Phi = phi.mean(axis=2)                                  # (N, L)
    Fk  = np.fft.fft(Phi, axis=1)
    Pk  = (Fk * Fk.conj()).real                             # (N, L)
    Gw  = np.fft.ifft(Pk, axis=1).real / L                  # cyclic auto-corr
    return Gw


def step_meff(Gw_mean: np.ndarray) -> np.ndarray:
    L  = Gw_mean.size
    me = np.full(L, np.nan)
    for t in range(1, L - 1):
        num = (Gw_mean[t - 1] + Gw_mean[t + 1]) / (2.0 * Gw_mean[t])
        if num >= 1.0:
            me[t] = np.arccosh(num)
    return me


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze_one(kappa_str: str, label: str):
    print(f"\n[{label}] loading {kappa_str} …")
    cfgs  = load_cfgs(kappa_str)
    N     = cfgs.shape[0]
    V     = L * L
    print(f"   N = {N}, shape = {cfgs.shape}")

    m_cfg = cfgs.mean(axis=(1, 2))
    mag   = magnetic_scalars(m_cfg, V)

    print("   computing per-cfg G(k) …")
    Gk    = per_cfg_pk(cfgs)
    KH2, DIAG = lattice_kh2_grid(L)
    mask  = (DIAG <= 0.51) & (KH2 > 1e-10)
    k_c, G_pc = radial_average_per_cfg(Gk, KH2, mask, n_bins=22)
    G_mean, G_err = bootstrap_mean_err(G_pc)
    fit = fit_xi(k_c, G_pc, n_low=5)

    # χ from G(k=0): G(0) = |Σφ|²/V → V·⟨φ²⟩_global with mean kept.
    # We use the connected χ from m_cfg above (which subtracts ⟨|m|⟩²).
    chi_from_G0 = float(Gk[:, 0, 0].mean())

    print("   computing wall correlator …")
    Gw_pc = wall_correlator_per_cfg(cfgs)
    Gw_mean, Gw_err = bootstrap_mean_err(Gw_pc)
    me = step_meff(Gw_mean)

    return dict(
        kappa=kappa_str, label=label, N=N,
        mag=mag, chi_G0=chi_from_G0,
        k_centres=k_c, G_mean=G_mean, G_err=G_err,
        fit=fit,
        Gw_mean=Gw_mean, Gw_err=Gw_err, meff=me,
    )


def write_summary(results, path: Path):
    lines = []
    lines.append(f"Comparison of L={L}, λ={LAMBDA} training cfgs\n")
    lines.append(f"{'observable':<30}  " + "  ".join(f"{r['label']:>26}" for r in results))
    lines.append("-" * (30 + 28 * len(results)))

    def row(name, fmt, getter):
        vals = "  ".join(f"{getter(r):>26{fmt}}" for r in results)
        lines.append(f"{name:<30}  {vals}")

    row("N_cfg",                    "d",  lambda r: r["N"])
    row("⟨|m|⟩",                    ".5f",lambda r: r["mag"]["abs_m"][0])
    row("  err",                    ".5f",lambda r: r["mag"]["abs_m"][1])
    row("χ_conn = V·(⟨m²⟩−⟨|m|⟩²)", ".2f",lambda r: r["mag"]["chi"][0])
    row("  err",                    ".2f",lambda r: r["mag"]["chi"][1])
    row("χ_full = G(k=0)",          ".2f",lambda r: r["chi_G0"])
    row("Binder U_4",               ".5f",lambda r: r["mag"]["U4"][0])
    row("  err",                    ".5f",lambda r: r["mag"]["U4"][1])
    row("m  (low-k fit)",           ".5f",lambda r: r["fit"]["m"])
    row("  err",                    ".5f",lambda r: r["fit"]["m_err"])
    row("ξ = 1/m",                  ".3f",lambda r: r["fit"]["xi"])
    row("  err",                    ".3f",lambda r: r["fit"]["xi_err"])
    row("ξ / L",                    ".4f",lambda r: r["fit"]["xi"] / L)

    txt = "\n".join(lines) + "\n"
    path.write_text(txt)
    print("\n" + txt)


def make_figure(results, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- (1) G(k̂) log-log ------------------------------------------------
    ax = axes[0]
    for r, c in zip(results, ["C0", "C3"]):
        ax.errorbar(r["k_centres"], r["G_mean"], yerr=r["G_err"],
                    fmt="o-", color=c, label=r["label"], lw=1.2, ms=4,
                    capsize=2)
    # reference power law k^-(2-η), η=0.25 → slope -1.75 (2D Ising)
    kmin = min(r["k_centres"].min() for r in results)
    kmax = max(r["k_centres"].max() for r in results)
    kref = np.geomspace(kmin, kmax, 50)
    g0   = max(r["G_mean"][0] for r in results)
    ax.plot(kref, g0 * (kref / kref[0])**(-1.75), "k--", lw=1.0, alpha=0.7,
            label=r"$k^{-(2-\eta)},\ \eta=0.25$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\hat k$"); ax.set_ylabel(r"$G(\hat k)$")
    ax.set_title("Radial momentum propagator")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # ---- (2) G_w(t) semi-log ---------------------------------------------
    ax = axes[1]
    t = np.arange(L)
    for r, c in zip(results, ["C0", "C3"]):
        ax.errorbar(t, r["Gw_mean"], yerr=r["Gw_err"],
                    fmt="o-", color=c, label=r["label"], lw=1.0, ms=3,
                    capsize=1.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$G_w(t)$")
    ax.set_title("Wall correlator (connected)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # ---- (3) step effective mass -----------------------------------------
    ax = axes[2]
    for r, c in zip(results, ["C0", "C3"]):
        ax.plot(t, r["meff"], "o-", color=c, label=r["label"], lw=1.0, ms=3)
        ax.axhline(r["fit"]["m"], color=c, ls=":", alpha=0.6,
                   label=fr"  $1/\xi={r['fit']['m']:.4f}$")
    ax.set_xlim(0, L // 2 + 4); ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$m_{\rm eff}(t)$")
    ax.set_title("Step effective mass  arccosh-type")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(rf"Scaling diagnostics — L=128, $\lambda$={LAMBDA}", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Saved figure: {path}")


def main():
    results = [analyze_one(k, lab) for k, lab in CASES]
    write_summary(results, OUT_DIR / "summary.txt")
    make_figure(results, OUT_DIR / "figure.png")


if __name__ == "__main__":
    main()
