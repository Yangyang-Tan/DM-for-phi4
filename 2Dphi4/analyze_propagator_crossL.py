"""
Momentum-space propagator G(k̂²) comparison for the cross-L experiment.

Convention (matches CorrelationUtils.jl in this repo):
    k̂² = Σᵢ 4 sin²(π nᵢ / L),         nᵢ ∈ [-L/2, L/2)
    G(k) = ⟨ |φ̃(k)|² ⟩ / V,           φ̃ = FFT(φ - ⟨φ⟩)
    Radial averaging: bin in k̂², drop modes whose
    diagonality  d = Σᵢ pᵢ⁴ / (Σᵢ pᵢ²)²  exceeds 0.51.

For each phase (k = 0.26, 0.2705, 0.28) we plot HMC L=64 vs DM L32→64 with
log and linear schedules, and (for completeness) include the axial G(k̂²_x).
Bootstrap errors are jackknife-style over n_boot resamples of configurations.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py


# ---------- I/O ----------------------------------------------------------------

def load_dm(path: Path) -> np.ndarray:
    arr = np.load(path)               # (L, L, N)
    return arr.transpose(2, 0, 1).astype(np.float64)


def load_hmc(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    if cfgs.ndim != 3:
        raise ValueError(f"unexpected ref shape {cfgs.shape}")
    sample_axis = int(np.argmax(cfgs.shape))
    if sample_axis != 0:
        cfgs = np.moveaxis(cfgs, sample_axis, 0)
    return cfgs


# ---------- lattice momentum --------------------------------------------------

def _signed_n(L: int) -> np.ndarray:
    n = np.arange(L)
    n = np.where(n > L // 2, n - L, n)
    return n


def lattice_kh2_grid(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (KH2, DIAG) of shape (L, L) with k̂² and diagonality."""
    n1 = _signed_n(L)
    p1 = 4 * np.sin(np.pi * n1 / L) ** 2  # along one axis
    P1 = p1[:, None] + np.zeros((L, L))
    P2 = np.zeros((L, L)) + p1[None, :]
    KH2 = P1 + P2
    sum_p4 = P1 * P1 + P2 * P2
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = np.where(KH2 > 1e-12, sum_p4 / KH2 ** 2, 0.0)
    return KH2, diag


# ---------- propagator --------------------------------------------------------

def per_config_pk(cfgs: np.ndarray, subtract_mean: bool = True) -> np.ndarray:
    """|φ̃(k)|² / V per configuration, shape (N, L, L)."""
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean() if subtract_mean else cfgs
    phi_k = np.fft.fft2(phi, axes=(1, 2))
    return (phi_k * phi_k.conj()).real / V


def radial_bin(values_per_cfg: np.ndarray, KH2: np.ndarray, mask: np.ndarray,
               n_bins: int = 16):
    """Bin |φ̃|² values radially in k̂². Returns (k_centres, mean, sem) over configs.

    `mask` is a boolean (L,L) selecting which modes participate.
    """
    N = values_per_cfg.shape[0]
    k_sel = KH2[mask]
    # log-spaced bins on positive k̂²
    pos = k_sel[k_sel > 1e-10]
    edges = np.geomspace(pos.min(), pos.max() * 1.001, n_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])

    # per-bin per-config mean
    nb = len(centres)
    means_per_cfg = np.zeros((N, nb))
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        in_bin = (k_sel >= edges[b]) & (k_sel < edges[b + 1])
        if not np.any(in_bin):
            continue
        counts[b] = in_bin.sum()
        # gather these mode values across configs
        vals = values_per_cfg[:, mask][:, in_bin]
        means_per_cfg[:, b] = vals.mean(axis=1)

    valid = counts > 0
    centres = centres[valid]
    means_per_cfg = means_per_cfg[:, valid]

    # bootstrap over configs
    rng = np.random.default_rng(0)
    n_boot = 200
    boots = np.empty((n_boot, means_per_cfg.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = means_per_cfg[idx].mean(axis=0)
    return centres, boots.mean(axis=0), boots.std(axis=0)


def axial_bin(values_per_cfg: np.ndarray, L: int):
    """Take G along axial direction (n_y=0), return (k̂², G, sem) for n_x = 1..L/2."""
    N = values_per_cfg.shape[0]
    n_x = np.arange(1, L // 2 + 1)
    k2 = 4 * np.sin(np.pi * n_x / L) ** 2  # n_y=0 → k̂²= 4 sin²(π n_x / L)
    g_per_cfg = values_per_cfg[:, n_x, 0]  # (N, len(n_x))
    rng = np.random.default_rng(0)
    n_boot = 200
    boots = np.empty((n_boot, g_per_cfg.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k2, boots.mean(axis=0), boots.std(axis=0)


# ---------- main --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L_train", type=int, default=32)
    p.add_argument("--L_sample", type=int, default=64)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--method", type=str, default="em")
    p.add_argument("--n_bins", type=int, default=16)
    p.add_argument("--max_diagonality", type=float, default=0.51)
    p.add_argument("--out", type=str, default="crossL_logs/crossL_propagator.pdf")
    args = p.parse_args()

    cases = [
        ("0.26",   "4999", "symmetric"),
        ("0.2705", "9999", "near-critical"),
        ("0.28",   "4999", "broken"),
    ]
    schedules = [("log",    "C3", "s"),
                 ("linear", "C2", "^")]

    L = args.L_sample
    KH2, DIAG = lattice_kh2_grid(L)
    radial_mask = (DIAG <= args.max_diagonality) & (KH2 > 1e-10)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    for col, (k, ep, label) in enumerate(cases):
        run_dir = Path(f"phi4_L{args.L_train}_k{k}_l{args.l}_ncsnpp")
        ref_path = Path("trainingdata") / f"cfgs_wolff_fahmc_k={k}_l={args.l}_{L}^2.jld2"

        ref = load_hmc(ref_path)
        if ref.shape[0] > 8192:
            ref = ref[:8192]
        pk_ref = per_config_pk(ref)

        dm = {}
        for sched, _, _ in schedules:
            tag = f"crossL_train{args.L_train}_sample{L}_{args.method}_{sched}_steps{args.num_steps}_ep{ep}"
            f = run_dir / "data_crossL" / f"samples_{tag}.npy"
            if not f.exists():
                print(f"[WARN] missing {f}")
                continue
            d = load_dm(f)
            dm[sched] = d
            print(f"[{label} k={k}]  {sched}: N={d.shape[0]}")

        # ---- Row 0: radial G(k̂²) ----
        ax = axes[0, col]
        kc, gm, ge = radial_bin(pk_ref, KH2, radial_mask, n_bins=args.n_bins)
        ax.errorbar(kc, gm, yerr=ge, fmt="o-", color="C0", lw=1.6, ms=4,
                    label=f"HMC (N={ref.shape[0]})", capsize=2)
        for sched, color, marker in schedules:
            if sched not in dm: continue
            pk_d = per_config_pk(dm[sched])
            kcd, gmd, ged = radial_bin(pk_d, KH2, radial_mask, n_bins=args.n_bins)
            ax.errorbar(kcd, gmd, yerr=ged, fmt=marker + "-", color=color, lw=1.4, ms=4,
                        label=f"DM[{sched}]", capsize=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\hat k^2$"); ax.set_ylabel(r"$G(\hat k^2)$")
        ax.set_title(f"{label} (k={k}) — radial")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

        # ---- Row 1: axial G(k̂²_x) ----
        ax = axes[1, col]
        k2_x, gm_x, ge_x = axial_bin(pk_ref, L)
        ax.errorbar(k2_x, gm_x, yerr=ge_x, fmt="o-", color="C0", lw=1.6, ms=4,
                    label="HMC", capsize=2)
        for sched, color, marker in schedules:
            if sched not in dm: continue
            pk_d = per_config_pk(dm[sched])
            k2d, gmd, ged = axial_bin(pk_d, L)
            ax.errorbar(k2d, gmd, yerr=ged, fmt=marker + "-", color=color, lw=1.4, ms=4,
                        label=f"DM[{sched}]", capsize=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\hat k^2_x$"); ax.set_ylabel(r"$G(\hat k^2_x)$")
        ax.set_title(f"{label} (k={k}) — axial (n_y=0)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Momentum-space propagator: train L={args.L_train} → sample L={L},  "
        f"NCSNpp,  steps={args.num_steps}",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
