"""
Compare HMC vs DM(EM, log) vs DM-MALA-corrected, in momentum space (axial G(k̂²_x)).

The MALA-corrected variant runs Phase 1 EM reverse SDE then Phase 2 Metropolis-
adjusted Langevin with the *true* L=64 phi^4 action — this should pull the IR
modes back toward HMC even though the score function only saw L=32 physics.

For each κ ∈ {0.2705, 0.28} we expect:
- κ=0.2705: large IR deficit (G/HMC≈0.63 at k̂²_x=0.0096) should partially close.
- κ=0.28:   uniform +5% bias should partially flatten.
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


def axial_G_with_err(pk: np.ndarray, L: int):
    n_x = np.arange(1, L // 2 + 1)
    k2 = 4 * np.sin(np.pi * n_x / L) ** 2
    g_per_cfg = pk[:, n_x, 0]
    rng = np.random.default_rng(0)
    n_boot = 200
    boots = np.empty((n_boot, g_per_cfg.shape[1]))
    N = pk.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k2, boots.mean(axis=0), boots.std(axis=0)


def main():
    cases = [("0.2705", "9999", "near-critical"),
             ("0.28",   "4999", "broken")]
    L = 64

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rows = []  # numerical table

    for col, (k, ep, label) in enumerate(cases):
        run_dir = Path(f"phi4_L32_k{k}_l0.022_ncsnpp/data_crossL")
        ref = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_64^2.jld2"))[:8192]

        runs = {}
        f_em = run_dir / f"samples_crossL_train32_sample64_em_log_steps2000_ep{ep}.npy"
        f_mala = run_dir / f"samples_crossL_train32_sample64_mala_log_steps1000_ep{ep}_tmh0.01_mh200_normref.npy"
        runs["DM-EM(log)"] = load_dm(f_em)
        runs["DM-MALA"]   = load_dm(f_mala)

        ax = axes[col]
        k2_r, g_r, e_r = axial_G_with_err(per_config_pk(ref), L)
        ax.errorbar(k2_r, g_r, yerr=e_r, fmt="o-", color="C0", lw=1.7, ms=5,
                    label=f"HMC (N={ref.shape[0]})", capsize=2)
        for name, color, marker in [("DM-EM(log)", "C3", "s"),
                                     ("DM-MALA",   "C2", "^")]:
            d = runs[name]
            k2_d, g_d, e_d = axial_G_with_err(per_config_pk(d), L)
            ax.errorbar(k2_d, g_d, yerr=e_d, fmt=marker + "-", color=color, lw=1.4, ms=5,
                        label=f"{name} (N={d.shape[0]})", capsize=2)

            for n_x_show in (1, 2, 3, 5, 10):
                idx = n_x_show - 1
                rows.append((label, k, name, k2_d[idx], g_d[idx], g_r[idx], g_d[idx] / g_r[idx]))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\hat k_x^2$"); ax.set_ylabel(r"$G(\hat k_x^2)$")
        ax.set_title(f"{label} (k={k})")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Cross-L (L=32 → L=64): MALA correction with true L=64 phi⁴ action",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    out = Path("crossL_logs/crossL_mala_compare.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print summary table
    print(f"\n{'phase':>14}  {'κ':>7}   {'method':>14}   {'k̂²_x':>8}  {'G_DM':>10}  {'G_HMC':>10}  ratio")
    print("-" * 92)
    for r in rows:
        print(f"{r[0]:>14}  {r[1]:>7}   {r[2]:>14}   {r[3]:8.4f}  {r[4]:10.4f}  {r[5]:10.4f}  {r[6]:5.3f}")


if __name__ == "__main__":
    main()
