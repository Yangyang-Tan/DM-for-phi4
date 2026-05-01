"""
Plot cross-L IR ratio (diagonal G(k_min)/HMC) as a function of training epoch.
Uses the linear-2000 cross-L samples generated at multiple ckpts.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dm(p: Path) -> np.ndarray:
    return np.load(p).transpose(2, 0, 1).astype(np.float64)


def load_hmc(p: Path) -> np.ndarray:
    with h5py.File(p, "r") as f:
        c = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(c.shape))
    if sa != 0:
        c = np.moveaxis(c, sa, 0)
    return c


def G_diag_kmin(cfgs: np.ndarray, n_boot: int = 200, seed: int = 0):
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V
    g_per_cfg = pk[:, 1, 1]  # n=(1,1) diagonal mode
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean()
    return boots.mean(), boots.std()


def main():
    L = 128
    k = "0.2705"
    run = Path(f"runs/phi4_Lmulti8-16-32-64_k{k}_l0.022_ncsnpp")

    ref = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2"))[:8192]
    G_hmc, E_hmc = G_diag_kmin(ref)
    print(f"HMC G(k_min, diagonal) = {G_hmc:.3f} ± {E_hmc:.3f}")

    # epscan_linear2000_ep<N> tag for the new scan; older runs use multiL64_no_lcond_..._ep<N>
    candidates = [
        # (epoch, tag_template)
        (201,  "epscan_linear2000_ep201"),
        (515,  "epscan_linear2000_ep515"),
        (1124, "epscan_linear2000_ep1124"),
        (2099, "epscan_linear2000_ep2099"),
        (3352, "epscan_linear2000_ep3352"),
        (5355, "epscan_linear2000_ep5355"),
        (8554, "multiL64_no_lcond_sample128_em_linear_steps2000_ep8554"),
        (10000,"multiL64_no_lcond_sample128_em_linear_steps2000_ep10000"),
    ]

    rows = []
    print(f"\n{'epoch':>6}  {'G(k_min)':>16}  {'ratio':>8}")
    for ep, tag in candidates:
        f = run / "data_crossL" / f"samples_{tag}.npy"
        if not f.exists():
            print(f"  {ep:>6}  [missing]  {f}")
            continue
        d = load_dm(f)
        G, E = G_diag_kmin(d)
        ratio = G / G_hmc
        ratio_err = E / G_hmc
        print(f"  {ep:>6}  {G:8.3f}±{E:6.3f}  {ratio:.4f}±{ratio_err:.4f}")
        rows.append((ep, G, E, ratio, ratio_err))

    if not rows:
        print("No data!")
        return

    rows = sorted(rows, key=lambda r: r[0])
    eps = np.array([r[0] for r in rows])
    G   = np.array([r[1] for r in rows])
    E   = np.array([r[2] for r in rows])
    ratio = np.array([r[3] for r in rows])
    ratio_err = np.array([r[4] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.errorbar(eps, ratio, yerr=ratio_err, fmt="o-", color="C2", lw=1.6, ms=7,
                capsize=3, label="DM linear 2000-step")
    ax.axhline(1.0, color="C0", ls="--", lw=1.4, alpha=0.8, label="HMC reference")
    best = np.argmax(ratio)
    ax.axvline(eps[best], color="C3", ls=":", lw=1.5,
               label=f"best @ ep={eps[best]} (ratio={ratio[best]:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"$G_{\rm DM}(k_{\rm min},\,{\rm diag})\,/\,G_{\rm HMC}$")
    ax.set_title("Cross-L IR ratio vs epoch  (L=128, diagonal lowest mode)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.errorbar(eps, G, yerr=E, fmt="o-", color="C2", lw=1.6, ms=7,
                capsize=3, label="DM linear 2000-step")
    ax.axhspan(G_hmc - E_hmc, G_hmc + E_hmc, color="C0", alpha=0.2,
               label=f"HMC ±1σ ({G_hmc:.1f}±{E_hmc:.1f})")
    ax.axhline(G_hmc, color="C0", ls="--", lw=1.4, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"$G(k_{\rm min},\,{\rm diag})$")
    ax.set_title("Absolute G(k_min) vs epoch")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)

    fig.suptitle(
        "Optimal epoch search (cross-L L=128 from multi-L L∈[8,16,32,64] σ=360 ckpt)",
        y=1.01, fontsize=13)
    plt.tight_layout()
    out = Path("results/crossL/optimal_epoch_scan.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")
    print(f"Best epoch: {eps[best]} with ratio = {ratio[best]:.4f}±{ratio_err[best]:.4f}")


if __name__ == "__main__":
    main()
