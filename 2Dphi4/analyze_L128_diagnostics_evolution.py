"""
Epoch-evolution of three diagnostic quantities for the L=128 phi^4 sweeps:
  1. Per-mode Gaussian KL:  D_k = ½(r - 1 - log r),  r = G_train / G_gen
  2. Statistical z-score:   z_k = (G_gen - G_train) / √(σ²_gen + σ²_train)
  3. Phase-space weighted:  w_k = k^(D-1) · (G_gen - G_train),  D=2

For each (k, method): produces a 3×2-panel figure:
  rows    = KL / |z| / |w|
  col 0   = diagnostic vs epoch at 5 representative |k| bins (IR→UV)
  col 1   = diagnostic vs |k|, all epochs overlaid, color-coded by epoch

Reuses propagators cached by analyze_L128_sweep.py in
  sigma_comparison_L128/L128_k{k}_sigma{σ}/prop_cache/*.npz
"""
import os, glob
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = "/data/tyywork/DM/2Dphi4"
OUT_ROOT = f"{ROOT}/sigma_comparison_L128"

CFG = [
    dict(k=0.2705, sigma=450, phase="near-critical"),
    dict(k=0.28,   sigma=640, phase="ordered"),
]
METHODS = ["em", "ode"]
EPOCHS = ["0001","0002","0003","0005","0009","0016","0028","0045","0079",
          "0138","0242","0422","0739","1291","2257","3593","6280","10000"]


def load_cached(cache_dir, label):
    d = np.load(f"{cache_dir}/{label}.npz")
    return d["k"], d["G"], d["Ge"]


def per_mode_kl(G_gen, G_train, eps=1e-30):
    Gg = np.maximum(G_gen, eps); Gt = np.maximum(G_train, eps)
    r = Gt / Gg
    return 0.5 * (r - 1 - np.log(r))


def zscore(G_gen, Ge_gen, G_train, Ge_train, eps=1e-30):
    denom = np.sqrt(Ge_gen**2 + Ge_train**2)
    denom = np.maximum(denom, eps)
    return (G_gen - G_train) / denom


def phase_space_weighted(k_vals, G_gen, G_train, D=2):
    # w_k = k^(D-1) * (G_gen - G_train)
    return (k_vals ** (D - 1)) * (G_gen - G_train)


def analyze_one(k, sigma, method):
    subdir = f"{OUT_ROOT}/L128_k{k}_sigma{sigma}"
    cache_dir = f"{subdir}/prop_cache"
    assert os.path.isdir(cache_dir), f"missing cache: {cache_dir}"

    kv, Gt, Gte = load_cached(cache_dir, "train")
    nz = kv > 1e-8
    k_nz = kv[nz]
    Gt_nz, Gte_nz = Gt[nz], Gte[nz]
    nb = len(k_nz)

    # pick 5 representative bins: IR, low-mid, mid, high-mid, UV
    idx_rep = np.array([0, nb // 6, nb // 3, nb // 2, 2 * nb // 3, nb - 2])
    idx_rep = np.unique(idx_rep)

    results = OrderedDict()
    for ep in EPOCHS:
        label = f"{method}_ep{ep}"
        path = f"{cache_dir}/{label}.npz"
        if not os.path.isfile(path):
            continue
        _, Gg, Gge = load_cached(cache_dir, label)
        Gg_nz, Gge_nz = Gg[nz], Gge[nz]
        results[ep] = dict(
            Gg=Gg_nz, Gge=Gge_nz,
            Dk=per_mode_kl(Gg_nz, Gt_nz),
            zk=zscore(Gg_nz, Gge_nz, Gt_nz, Gte_nz),
            wk=phase_space_weighted(k_nz, Gg_nz, Gt_nz, D=2),
        )

    # Figure: 3 rows × 2 cols
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    ep_nums = np.array([int(e) for e in results.keys()])
    ep_colors = mpl.colormaps["plasma"](
        (np.log(ep_nums) - np.log(ep_nums.min())) /
        (np.log(ep_nums.max()) - np.log(ep_nums.min())))

    # column-0 color: by |k| bin (IR→UV, cool→warm)
    kbin_colors = mpl.colormaps["viridis"](np.linspace(0, 1, len(idx_rep)))

    # ---- Row 0: KL ----
    ax = axes[0, 0]
    for ci, ki in enumerate(idx_rep):
        ys = [results[ep]["Dk"][ki] for ep in results]
        ax.plot(ep_nums, ys, "o-", ms=4, color=kbin_colors[ci],
                label=f"|k|={k_nz[ki]:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("per-mode KL  D_k")
    ax.set_title("KL vs epoch (5 k bins)"); ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for j, ep in enumerate(results):
        ax.plot(k_nz, results[ep]["Dk"], "-", color=ep_colors[j], alpha=0.8, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("D_k")
    ax.set_title("KL vs |k| (purple=early → yellow=late)")
    ax.grid(alpha=0.3)

    # ---- Row 1: |z| ----
    ax = axes[1, 0]
    for ci, ki in enumerate(idx_rep):
        ys = [abs(results[ep]["zk"][ki]) for ep in results]
        ax.plot(ep_nums, ys, "o-", ms=4, color=kbin_colors[ci],
                label=f"|k|={k_nz[ki]:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("|z_k|")
    ax.set_title("|z-score| vs epoch"); ax.legend(fontsize=8, ncol=2)
    ax.axhline(2.0, linestyle="--", color="gray", lw=0.8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for j, ep in enumerate(results):
        ax.plot(k_nz, results[ep]["zk"], "-", color=ep_colors[j], alpha=0.8, lw=1)
    ax.axhline(0.0, linestyle="--", color="gray")
    for y in (-2, 2):
        ax.axhline(y, linestyle=":", color="gray", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("z_k")
    ax.set_title("z-score vs |k|")
    ax.grid(alpha=0.3)

    # ---- Row 2: w_k (phase-space weighted) ----
    ax = axes[2, 0]
    for ci, ki in enumerate(idx_rep):
        ys = [results[ep]["wk"][ki] for ep in results]
        ax.plot(ep_nums, ys, "o-", ms=4, color=kbin_colors[ci],
                label=f"|k|={k_nz[ki]:.3f}")
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("w_k = k·(G_DM - G_train)")
    ax.set_title("phase-space weighted Δ vs epoch")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    for j, ep in enumerate(results):
        ax.plot(k_nz, results[ep]["wk"], "-", color=ep_colors[j], alpha=0.8, lw=1)
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("w_k")
    ax.set_title("w_k vs |k|")
    ax.grid(alpha=0.3)

    # colorbar for epoch on all col=1 plots (once)
    sm = mpl.cm.ScalarMappable(
        cmap=mpl.colormaps["plasma"],
        norm=mpl.colors.LogNorm(vmin=ep_nums.min()+1, vmax=ep_nums.max()))
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(sm, cax=cbar_ax); cb.set_label("epoch")

    plt.suptitle(
        f"L=128  k={k}  σ={sigma}  method={method}  (UV/IR diagnostics vs epoch)",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])

    outpath = f"{subdir}/diagnostics_evolution_{method}.png"
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outpath}")

    # ---- compact summary: integrated metrics vs epoch ----
    tot = OrderedDict()
    for ep in results:
        tot[ep] = dict(
            KL_sum=float(results[ep]["Dk"].sum()),
            z_absmax=float(np.abs(results[ep]["zk"]).max()),
            z_rmse=float(np.sqrt(np.mean(results[ep]["zk"]**2))),
            w_integral=float((results[ep]["wk"] * np.gradient(k_nz)).sum()),
            # IR/UV split
            KL_IR=float(results[ep]["Dk"][k_nz < 0.5].sum()),
            KL_UV=float(results[ep]["Dk"][k_nz > 1.5].sum()),
            wk_abs_IR=float(np.abs(results[ep]["wk"][k_nz < 0.5]).mean()),
            wk_abs_UV=float(np.abs(results[ep]["wk"][k_nz > 1.5]).mean()),
        )
    return tot


def plot_integrated(all_summaries, outpath):
    """Plot integrated KL and IR-vs-UV split across kappas/methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    markers = dict(em="o", ode="^")
    colors  = dict(em="C0", ode="C3")

    # Panel 0: total KL vs epoch
    ax = axes[0]
    for (k, method), tot in all_summaries.items():
        eps = np.array([int(e) for e in tot.keys()])
        ys = [tot[e]["KL_sum"] for e in tot]
        ax.plot(eps, ys, marker=markers[method], ms=4, ls="-",
                label=f"k={k} {method}", color=colors[method],
                alpha=1.0 if k == 0.2705 else 0.5,
                mfc="white" if k == 0.28 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("Σ D_k (integrated KL)")
    ax.set_title("total Gaussian KL"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 1: IR vs UV KL split
    ax = axes[1]
    for (k, method), tot in all_summaries.items():
        eps = np.array([int(e) for e in tot.keys()])
        ir = [tot[e]["KL_IR"] for e in tot]
        uv = [tot[e]["KL_UV"] for e in tot]
        label_base = f"k={k} {method}"
        ax.plot(eps, ir, marker=markers[method], ms=4, ls="-",
                color=colors[method], alpha=0.7 if k == 0.28 else 1.0,
                label=f"{label_base} IR (|k|<0.5)")
        ax.plot(eps, uv, marker=markers[method], ms=4, ls="--",
                color=colors[method], alpha=0.7 if k == 0.28 else 1.0,
                label=f"{label_base} UV (|k|>1.5)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("Σ D_k (IR / UV subset)")
    ax.set_title("KL IR vs UV"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 2: |z|_rmse vs epoch
    ax = axes[2]
    for (k, method), tot in all_summaries.items():
        eps = np.array([int(e) for e in tot.keys()])
        ys = [tot[e]["z_rmse"] for e in tot]
        ax.plot(eps, ys, marker=markers[method], ms=4, ls="-",
                color=colors[method], alpha=0.7 if k == 0.28 else 1.0,
                label=f"k={k} {method}")
    ax.axhline(1.0, linestyle=":", color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("z_rmse = √⟨z_k²⟩")
    ax.set_title("statistical z-score RMSE"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Integrated diagnostics vs epoch  (L=128 2D phi⁴)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved summary -> {outpath}")


if __name__ == "__main__":
    all_summaries = {}
    for cfg in CFG:
        for method in METHODS:
            print(f"\n-- k={cfg['k']} σ={cfg['sigma']} method={method} --")
            tot = analyze_one(cfg["k"], cfg["sigma"], method)
            all_summaries[(cfg["k"], method)] = tot

    plot_integrated(all_summaries, f"{OUT_ROOT}/diagnostics_integrated_all.png")

    # also print a table
    print("\n" + "="*78)
    print("  INTEGRATED METRICS AT ep=10000")
    print("="*78)
    print(f"{'k':>7}  {'method':>6}  {'ΣD_k':>8}  {'KL_IR':>8}  {'KL_UV':>8}  "
          f"{'z_rmse':>7}  {'|z|_max':>7}  {'|w|_IR':>9}  {'|w|_UV':>9}")
    for (k, method), tot in all_summaries.items():
        last = tot["10000"]
        print(f"{k:>7.4f}  {method:>6}  {last['KL_sum']:>8.4f}  "
              f"{last['KL_IR']:>8.4f}  {last['KL_UV']:>8.4f}  "
              f"{last['z_rmse']:>7.2f}  {last['z_absmax']:>7.2f}  "
              f"{last['wk_abs_IR']:>9.4f}  {last['wk_abs_UV']:>9.4f}")
