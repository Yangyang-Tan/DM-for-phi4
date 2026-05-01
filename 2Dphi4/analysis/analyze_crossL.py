"""
Compare L=32→L=64 cross-L DM samples to L=64 HMC reference data.

Overlays HMC reference, DM(log schedule), and DM(linear schedule) for three
kappa values (symmetric / near-critical / broken). Metrics:
  - per-site phi histogram (single-site marginal)
  - sample-mean magnetisation distribution
  - action density per site
  - radial 2-pt correlator (FFT-based)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dm_samples(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(f"unexpected shape {arr.shape} in {npy_path}")
    return arr.transpose(2, 0, 1).astype(np.float64)  # -> (N, L, L)


def load_hmc_reference(jld2_path: str) -> np.ndarray:
    with h5py.File(jld2_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    cfgs = np.asarray(cfgs, dtype=np.float64)
    if cfgs.ndim != 3:
        raise ValueError(f"unexpected ref shape {cfgs.shape}")
    sample_axis = int(np.argmax(cfgs.shape))
    if sample_axis != 0:
        cfgs = np.moveaxis(cfgs, sample_axis, 0)
    return cfgs


def action_per_site(phi: np.ndarray, k: float, l: float) -> np.ndarray:
    nb = np.roll(phi, 1, axis=1) + np.roll(phi, 1, axis=2)
    s = -2 * k * phi * nb + (1 - 2 * l) * phi**2 + l * phi**4
    V = phi.shape[1] * phi.shape[2]
    return s.sum(axis=(1, 2)) / V


def radial_corr(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, L, _ = phi.shape
    phi_c = phi - phi.mean()
    fk = np.fft.fft2(phi_c, axes=(1, 2))
    pk = (fk * fk.conj()).real / (L * L)
    cr = np.fft.ifft2(pk.mean(axis=0)).real
    yy, xx = np.indices((L, L))
    yy = np.minimum(yy, L - yy)
    xx = np.minimum(xx, L - xx)
    r = np.sqrt(xx**2 + yy**2).flatten()
    rmax = L // 2
    bins = np.arange(0, rmax + 1) - 0.5
    counts, _ = np.histogram(r, bins=bins)
    sums, _ = np.histogram(r, bins=bins, weights=cr.flatten())
    centres = 0.5 * (bins[:-1] + bins[1:])
    valid = counts > 0
    return centres[valid], (sums[valid] / counts[valid])


def fmt(x: np.ndarray) -> str:
    return f"{x.mean():.3f}±{x.std():.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L_train", type=int, default=32)
    p.add_argument("--L_sample", type=int, default=64)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--method", type=str, default="em")
    p.add_argument("--out", type=str, default="results/crossL/crossL_compare.pdf")
    args = p.parse_args()

    cases = [
        ("0.26",   "4999", "symmetric"),
        ("0.2705", "9999", "near-critical"),
        ("0.28",   "4999", "broken"),
    ]
    schedules = [("log", "C3", "s"), ("linear", "C2", "^")]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))

    for row, (k, ep, label) in enumerate(cases):
        kf = float(k)
        run_dir = Path(f"runs/phi4_L{args.L_train}_k{k}_l{args.l}_ncsnpp")
        ref_path = Path("trainingdata") / f"cfgs_wolff_fahmc_k={k}_l={args.l}_{args.L_sample}^2.jld2"
        ref = load_hmc_reference(str(ref_path))
        # cap reference for histogram fairness
        if ref.shape[0] > 8192:
            ref = ref[:8192]
        print(f"[{label} k={k}] REF N={ref.shape[0]} L={ref.shape[1]} S/V={fmt(action_per_site(ref, kf, args.l))}")

        dm_runs = {}
        for sched, _, _ in schedules:
            tag = f"crossL_train{args.L_train}_sample{args.L_sample}_{args.method}_{sched}_steps{args.num_steps}_ep{ep}"
            dm_path = run_dir / "data_crossL" / f"samples_{tag}.npy"
            if not dm_path.exists():
                print(f"   [WARN] missing {dm_path}")
                continue
            dm_runs[sched] = load_dm_samples(str(dm_path))
            print(f"   DM[{sched}] N={dm_runs[sched].shape[0]} S/V={fmt(action_per_site(dm_runs[sched], kf, args.l))}")

        # bin ranges driven by reference
        # 1) per-site phi
        ax = axes[row, 0]
        all_min = min(ref.min(), *(d.min() for d in dm_runs.values()))
        all_max = max(ref.max(), *(d.max() for d in dm_runs.values()))
        bins = np.linspace(all_min, all_max, 100)
        ax.hist(ref.ravel(), bins=bins, density=True, histtype="step", lw=2.0, color="C0", label="HMC L=64")
        for sched, color, _ in schedules:
            if sched in dm_runs:
                ax.hist(dm_runs[sched].ravel(), bins=bins, density=True, histtype="step", lw=1.5,
                        color=color, label=f"DM L32→64 [{sched}]")
        ax.set_title(f"{label} (k={k}): P(φ)")
        ax.set_xlabel("φ"); ax.set_ylabel("p"); ax.legend(fontsize=8)

        # 2) volume-mean magnetisation
        ax = axes[row, 1]
        m_ref = ref.mean(axis=(1, 2))
        m_runs = {s: d.mean(axis=(1, 2)) for s, d in dm_runs.items()}
        all_min = min(m_ref.min(), *(m.min() for m in m_runs.values()))
        all_max = max(m_ref.max(), *(m.max() for m in m_runs.values()))
        bins = np.linspace(all_min, all_max, 60)
        ax.hist(m_ref, bins=bins, density=True, histtype="step", lw=2.0, color="C0", label="HMC")
        for sched, color, _ in schedules:
            if sched in m_runs:
                ax.hist(m_runs[sched], bins=bins, density=True, histtype="step", lw=1.5,
                        color=color, label=f"DM[{sched}]")
        ax.set_title(f"{label}: P(<φ>_V)")
        ax.set_xlabel("<φ>_V"); ax.legend(fontsize=8)

        # 3) action density
        ax = axes[row, 2]
        s_ref = action_per_site(ref, kf, args.l)
        s_runs = {s: action_per_site(d, kf, args.l) for s, d in dm_runs.items()}
        all_min = min(s_ref.min(), *(s.min() for s in s_runs.values()))
        all_max = max(s_ref.max(), *(s.max() for s in s_runs.values()))
        bins = np.linspace(all_min, all_max, 60)
        ax.hist(s_ref, bins=bins, density=True, histtype="step", lw=2.0, color="C0", label=f"HMC ({fmt(s_ref)})")
        for sched, color, _ in schedules:
            if sched in s_runs:
                ax.hist(s_runs[sched], bins=bins, density=True, histtype="step", lw=1.5,
                        color=color, label=f"DM[{sched}] ({fmt(s_runs[sched])})")
        ax.set_title(f"{label}: S/V")
        ax.set_xlabel("S/V"); ax.legend(fontsize=7)

        # 4) connected radial correlator (log y, |C|)
        ax = axes[row, 3]
        r_ref, c_ref = radial_corr(ref)
        ax.semilogy(r_ref, np.maximum(np.abs(c_ref), 1e-6), "o-", color="C0", lw=1.6, ms=4, label="HMC")
        for sched, color, marker in schedules:
            if sched in dm_runs:
                r_d, c_d = radial_corr(dm_runs[sched])
                ax.semilogy(r_d, np.maximum(np.abs(c_d), 1e-6), marker + "-", color=color, lw=1.4, ms=4,
                            label=f"DM[{sched}]")
        ax.set_title(f"{label}: |G(r)|")
        ax.set_xlabel("r"); ax.set_ylabel("|G(r)|"); ax.legend(fontsize=8)

    fig.suptitle(
        f"Cross-L diffusion: trained on L={args.L_train}, sampled at L={args.L_sample}, "
        f"steps={args.num_steps}, method={args.method}",
        y=1.00, fontsize=14,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
