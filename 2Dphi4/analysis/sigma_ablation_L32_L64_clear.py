"""
Clean re-do of fig5_L32_vs_L64.png with explicit, unambiguous axis labels.

Original plot's x-axis "σ_max/D_max" was ambiguous — could not distinguish
σ_param/D_max_orig from std(t=1)/D_max_norm. Here we plot vs the SDE
parameter σ directly (no ratios), and overlay vertical reference lines for
both candidate rules:

  - "2D rule"  (CLAUDE.md):  std(t=1) = D_max_orig / 2
  - "3D rule"  (alternative): std(t=1) = D_max_orig
  - σ_param = D_max_orig (yet another candidate "rule")

Both L=32 and L=64 are sampled IN-DISTRIBUTION (separate σ-ablation runs
at each L), no cross-L OOD. κ=0.28 is used because it has the most complete
ablation grid at both L values.

Observable: cumulants of |M| (Z_2-symmetric magnetisation), since κ=0.28 is
in the broken phase (HMC ⟨|M|⟩ ≈ 0.7-0.8). Deviation % = (DM-HMC)/HMC × 100.
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.distance import pdist
from scipy.optimize import brentq


KAPPA = 0.28
LAMBDA = 0.022
EP = 10000
N_SUBSAMPLE = 2048    # for D_max measurement
SEED = 0


def std_t1(sigma):
    return math.sqrt((sigma ** 2 - 1.0) / (2.0 * math.log(sigma)))


def sigma_for_target_std(target):
    return brentq(lambda s: std_t1(s) - target, 1.001, 1.0e6)


def load_dm_samples(path):
    return np.load(path).transpose(2, 0, 1).astype(np.float64)


def load_hmc(path, max_n=8192):
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    sa = int(np.argmax(cfgs.shape))
    if cfgs.ndim == 3 and sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs[:max_n]


def measure_Dmax(hmc_cfgs):
    """D_max in BOTH original space and normalised [-1,1] space."""
    flat = hmc_cfgs.reshape(hmc_cfgs.shape[0], -1)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(flat.shape[0], min(N_SUBSAMPLE, flat.shape[0]), replace=False)
    Dmax_orig = float(pdist(flat[idx]).max())
    lo, hi = float(flat.min()), float(flat.max())
    nrm = ((flat[idx] - lo) / (hi - lo) - 0.5) * 2.0
    Dmax_norm = float(pdist(nrm).max())
    return Dmax_orig, Dmax_norm


def cumulants_absM(cfgs):
    """Raw 2nd and 4th moments of M = ⟨φ⟩_cfg (Z_2 symmetric so ⟨M⟩≈0).

    Using raw moments (not central / cumulants) because:
    - 4th-order cumulants need very large N to converge (subtractive cancellation)
    - In broken phase ⟨M⟩≈0 by Z_2 symmetry, so raw moments = central moments
    - Raw moments are O(1) and stable across the σ ablation grid
    """
    M = cfgs.mean(axis=(1, 2))
    k2 = (M ** 2).mean()      # ⟨M²⟩  (susceptibility-like)
    k4 = (M ** 4).mean()      # ⟨M⁴⟩
    return k2, k4


def collect(L, sigma_grid):
    Dmax_orig, Dmax_norm = measure_Dmax(
        load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={KAPPA}_l={LAMBDA}_{L}^2.jld2"))
    )
    hmc = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={KAPPA}_l={LAMBDA}_{L}^2.jld2"),
                   max_n=8192)
    k2_h, k4_h = cumulants_absM(hmc)
    print(f"\n=== L={L} κ={KAPPA} ===")
    print(f"  D_max_orig = {Dmax_orig:.3f}")
    print(f"  D_max_norm = {Dmax_norm:.3f}")
    print(f"  HMC: κ_2(|M|)={k2_h:.5f}  κ_4(|M|)={k4_h:.5e}")

    rows = []
    for sig in sigma_grid:
        d = Path(f"runs/phi4_L{L}_k{KAPPA}_l{LAMBDA}_ncsnpp_sigma{sig}/data")
        em_p = d / f"samples_em_steps2000_{EP:04d}.npy"
        ode_p = d / f"samples_ode_steps400_{EP:04d}.npy"
        if not em_p.exists() or not ode_p.exists():
            print(f"  σ={sig:>4}: missing samples — skip")
            continue
        em = load_dm_samples(em_p)
        ode = load_dm_samples(ode_p)
        k2_em, k4_em = cumulants_absM(em)
        k2_od, k4_od = cumulants_absM(ode)
        rows.append({
            "sigma": sig,
            "std_t1": std_t1(sig),
            "k2_em": k2_em, "k4_em": k4_em,
            "k2_ode": k2_od, "k4_ode": k4_od,
        })
        print(f"  σ={sig:>4} std(t=1)={std_t1(sig):>7.2f}  "
              f"EM:  κ_2={k2_em:.5f} ({100*(k2_em-k2_h)/k2_h:+5.2f}%)  "
              f"κ_4={k4_em:+.3e} ({100*(k4_em-k4_h)/k4_h:+6.2f}%)  "
              f"ODE: κ_2={k2_od:.5f} ({100*(k2_od-k2_h)/k2_h:+5.2f}%)  "
              f"κ_4={k4_od:+.3e} ({100*(k4_od-k4_h)/k4_h:+6.2f}%)")
    return Dmax_orig, Dmax_norm, k2_h, k4_h, rows


def main():
    res_32 = collect(L=32, sigma_grid=[15, 30, 60, 90, 180, 360])
    res_64 = collect(L=64, sigma_grid=[25, 50, 100, 150, 300, 600])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, obs_key, ylabel in [(axes[0], ("k2_em", "k2_ode", "k2_h"), r"$\langle M^2 \rangle$ deviation (%)"),
                                 (axes[1], ("k4_em", "k4_ode", "k4_h"), r"$\langle M^4 \rangle$ deviation (%)")]:
        em_key, ode_key, h_key = obs_key
        for (L, color, marker, ls), (Dmax_o, Dmax_n, k2_h, k4_h, rows) in [
            ((32, "C0", "s", "-"), res_32),
            ((64, "C1", "o", "--"), res_64),
        ]:
            href = k2_h if "k2" in em_key else k4_h
            x = np.array([r["sigma"] for r in rows])
            y_em = np.array([100*(r[em_key] - href)/href for r in rows])
            y_ode = np.array([100*(r[ode_key] - href)/href for r in rows])
            ax.plot(x, y_em, marker + ls, color=color, lw=1.5, ms=7,
                    label=f"L={L} EM")
            ax.plot(x, y_ode, marker + ls, color=color, lw=1.5, ms=7, mfc="none",
                    label=f"L={L} ODE")
            # Reference vertical lines for THIS L's rule predictions
            sig_2D = sigma_for_target_std(Dmax_o / 2)
            sig_3D = sigma_for_target_std(Dmax_o)
            sig_eq = Dmax_o   # σ_param = D_max_orig
            for sig, label, color2, lstyle in [
                (sig_2D, f"L={L}: σ such that std(t=1)=D_max_orig/2  (={sig_2D:.0f})",
                 color, ":"),
                (sig_3D, f"L={L}: σ such that std(t=1)=D_max_orig    (={sig_3D:.0f})",
                 color, "-."),
            ]:
                ax.axvline(sig, color=color2, ls=lstyle, alpha=0.5, lw=0.8)

        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"SDE parameter $\sigma$ (the `--sigma` flag value)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"σ ablation, in-distribution sampling at L=32 and L=64 (κ={KAPPA}, ep={EP})\n"
        r"Reference lines:  dotted = $\sigma$ predicted by 2D rule (std(t=1)=D_max_orig/2);  "
        r"dash-dot = 3D rule (std(t=1)=D_max_orig)",
        y=1.00, fontsize=11)

    fig.tight_layout()
    out = Path("results/sigma_ablation/L32_L64_kappa0.28_redo.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
