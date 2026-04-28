"""
L=128 phi^4 2D sweep analysis, CelebA-style.

For each k in {0.2705, 0.28}, method in {em, ode}:
  - Load 18 log-spaced epochs of 2048 samples each
  - Compute radial-averaged G(|k|) vs training-data propagator
  - Produce 4-panel end-of-training comparison at ep=10000 (like CelebA)
  - Produce multi-epoch evolution curves

Outputs under sigma_comparison_L128/L128_{k}_{method}/.
"""
import os, glob
from collections import OrderedDict
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = "/data/tyywork/DM/2Dphi4"
OUT_ROOT = f"{ROOT}/sigma_comparison_L128"
os.makedirs(OUT_ROOT, exist_ok=True)

# matches sample_phi4_sweep.py DEFAULT_EPOCHS
EPOCHS = ["0001","0002","0003","0005","0009","0016","0028","0045","0079",
          "0138","0242","0422","0739","1291","2257","3593","6280","10000"]

CFG = [
    dict(k=0.2705, sigma=450, norm_min=-4.4730, norm_max=4.7432, phase="near-critical"),
    dict(k=0.28,   sigma=640, norm_min=-4.6458, norm_max=4.7210, phase="ordered"),
]

# ---------------------------- loaders ---------------------------------
def load_train(k):
    path = f"{ROOT}/trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_128^2.jld2"
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    if cfgs.ndim == 3 and cfgs.shape[0] != cfgs.shape[1]:
        cfgs = cfgs.transpose(1, 2, 0)  # (N,L,L) -> (L,L,N)
    return cfgs

def load_gen(k, sigma, method, ep):
    steps = 2000 if method == "em" else 400
    path = (f"{ROOT}/phi4_L128_k{k}_l0.022_ncsnpp_sigma{sigma}"
            f"/data/samples_{method}_steps{steps}_epoch={ep}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path).astype(np.float32)

# ---------------------------- propagator -------------------------------
def build_bins(L=128, max_diag=0.51):
    ns = np.arange(L); ns = np.where(ns > L//2, ns - L, ns)
    KX, KY = np.meshgrid(ns, ns, indexing="ij")
    kh2 = 4*np.sin(np.pi*KX/L)**2 + 4*np.sin(np.pi*KY/L)**2
    p2x = 4*np.sin(np.pi*KX/L)**2
    p2y = 4*np.sin(np.pi*KY/L)**2
    sum_p2 = p2x + p2y; sum_p4 = p2x**2 + p2y**2
    diag = np.zeros_like(sum_p2); m0 = sum_p2 > 1e-10
    diag[m0] = sum_p4[m0] / sum_p2[m0]**2
    sel = (diag <= max_diag) | ~m0
    ksq_u = np.sort(np.unique(np.round(kh2[sel], 5)))
    k_vals = np.sqrt(ksq_u)
    bin_idx = [np.where(np.abs(kh2 - ks) < 1e-4) for ks in ksq_u]
    return k_vals, bin_idx

def compute_propagator(cfgs, bins=None, n_boot=200, seed=42):
    L = cfgs.shape[0]; N = cfgs.shape[-1]; V = L * L
    if bins is None:
        k_vals, bin_idx = build_bins(L)
    else:
        k_vals, bin_idx = bins
    nb = len(k_vals)
    G_rad = np.zeros((nb, N), dtype=np.float64)
    for i in range(N):
        cfg = cfgs[..., i].astype(np.float64)
        cfg = cfg - cfg.mean()
        ps = np.abs(np.fft.fft2(cfg))**2 / V
        for j, idx in enumerate(bin_idx):
            G_rad[j, i] = ps[idx].mean()
    G_mean = G_rad.mean(axis=1)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, nb))
    for b in range(n_boot):
        ii = rng.integers(0, N, size=N)
        boots[b] = G_rad[:, ii].mean(axis=1)
    return k_vals, G_mean, boots.std(axis=0)

def per_mode_kl(G_gen, G_train, eps=1e-30):
    Gg = np.maximum(G_gen, eps); Gt = np.maximum(G_train, eps)
    r = Gt / Gg
    return 0.5 * (r - 1 - np.log(r))

# ---------------------------- caching ----------------------------------
def propagator_cached(cfgs, label, cache_dir):
    cache = f"{cache_dir}/{label}.npz"
    if os.path.isfile(cache):
        d = np.load(cache)
        return d["k"], d["G"], d["Ge"]
    k_vals, G, Ge = compute_propagator(cfgs)
    np.savez(cache, k=k_vals, G=G, Ge=Ge)
    return k_vals, G, Ge

# ---------------------------- 4-panel (late epoch) --------------------
def plot_4panel(k_vals, Gt, Gte, Gg, Gge, title, outpath):
    nz = k_vals > 1e-8
    kv = k_vals[nz]; Gt, Gte = Gt[nz], Gte[nz]; Gg, Gge = Gg[nz], Gge[nz]
    ratio = Gg / Gt
    ratio_err = ratio * np.sqrt((Gge/Gg)**2 + (Gte/Gt)**2)
    z = (Gg - Gt) / np.sqrt(Gge**2 + Gte**2 + 1e-30)
    kl = per_mode_kl(Gg, Gt)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))

    ax = axes[0, 0]
    ax.errorbar(kv, Gt, yerr=Gte, fmt="o", ms=3, label="train", color="black", capsize=2)
    ax.errorbar(kv, Gg, yerr=Gge, fmt="s", ms=3, label="DM", color="C0", capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("G(k)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(kv, ratio, yerr=ratio_err, fmt="s", ms=3, color="C0", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("G_DM / G_train")
    ax.set_title("ratio"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(kv, z, "s", ms=3, color="C0")
    for y in (-2, 0, 2):
        ax.axhline(y, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("z-score  (G_DM - G_train) / √(σ²)")
    ax.set_title("z-score per bin"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(kv, kl, "s", ms=3, color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("D_k")
    ax.set_title("per-mode Gaussian KL"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    return dict(
        max_absratio=float(np.abs(ratio-1).max()),
        mean_absratio=float(np.abs(ratio-1).mean()),
        max_absz=float(np.abs(z).max()),
        mean_absz=float(np.abs(z).mean()),
        sum_kl=float(kl.sum()),
    )

# ---------------------------- evolution plots -------------------------
def plot_evolution(all_results, k_vals, Gt, k_label, method, outpath,
                   k_highlights=None):
    """all_results: dict ep -> (Gg, Gge)."""
    if k_highlights is None:
        # choose IR/mid/UV representative bins
        nb = len(k_vals)
        nz_first = np.where(k_vals > 1e-8)[0][0]
        k_highlights = [nz_first, nz_first + 3, nb//4, nb//2, nb*3//4, nb-2]
    eps_sorted = sorted(all_results, key=int)
    ep_nums = np.array([int(e) for e in eps_sorted])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: G(k)/G_train at several representative k values vs epoch
    ax = axes[0]
    cmap = mpl.colormaps.get_cmap("viridis")
    for i, ki in enumerate(k_highlights):
        if ki == 0 or k_vals[ki] < 1e-8:
            continue
        ratios = [all_results[ep][0][ki] / Gt[ki] for ep in eps_sorted]
        errs   = [all_results[ep][1][ki] / Gt[ki] for ep in eps_sorted]
        c = cmap(i / max(1, len(k_highlights) - 1))
        ax.errorbar(ep_nums, ratios, yerr=errs, fmt="o-", ms=4, color=c,
                    label=f"|k|={k_vals[ki]:.3f}", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("G_DM(k) / G_train(k)")
    ax.set_title(f"propagator evolution  (k={k_label}, {method})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Right: ratio curves vs |k|, colored by epoch
    ax = axes[1]
    cmap2 = mpl.colormaps.get_cmap("plasma")
    for j, ep in enumerate(eps_sorted):
        Gg, Gge = all_results[ep]
        nz = k_vals > 1e-8
        ratio = Gg[nz] / Gt[nz]
        c = cmap2(j / max(1, len(eps_sorted) - 1))
        ax.plot(k_vals[nz], ratio, "-", color=c, alpha=0.7)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("G_DM / G_train")
    ax.set_title(f"ratio across epochs  (first=purple → last=yellow)")
    ax.grid(alpha=0.3)
    # colorbar for epoch
    sm = mpl.cm.ScalarMappable(cmap=cmap2,
        norm=mpl.colors.LogNorm(vmin=ep_nums.min()+1, vmax=ep_nums.max()))
    cb = plt.colorbar(sm, ax=ax); cb.set_label("epoch")

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------- em vs ode -------------------------------
def plot_em_vs_ode(k_vals, Gt, em_res, ode_res, k_label, outpath):
    """Compare em and ode at the last epoch."""
    ep = "10000"
    Gem, Gem_e = em_res[ep]; Gode, Gode_e = ode_res[ep]
    nz = k_vals > 1e-8
    kv = k_vals[nz]
    ratio_em = Gem[nz] / Gt[nz]; ratio_ode = Gode[nz] / Gt[nz]
    re_err = ratio_em * np.sqrt((Gem_e[nz]/Gem[nz])**2 + 1e-12)
    ro_err = ratio_ode * np.sqrt((Gode_e[nz]/Gode[nz])**2 + 1e-12)
    # em vs ode diff (same IC, same seed)
    em_ode_diff = (Gem[nz] - Gode[nz]) / ((Gem[nz] + Gode[nz]) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.errorbar(kv, ratio_em, yerr=re_err, fmt="s", ms=3, label="em (SDE)",
                color="C0", capsize=2)
    ax.errorbar(kv, ratio_ode, yerr=ro_err, fmt="^", ms=3, label="ode (DPM-2)",
                color="C3", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("G_DM / G_train")
    ax.set_title(f"em vs ode, ep=10000 (k={k_label})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(kv, em_ode_diff * 100, "o", ms=3, color="C4")
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|")
    ax.set_ylabel("(G_em - G_ode) / ((G_em+G_ode)/2) × 100%")
    ax.set_title(f"em - ode (same IC), ep=10000  k={k_label}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------- main -----------------------------------
def analyze_kappa(cfg):
    k, sigma = cfg["k"], cfg["sigma"]
    subdir = f"{OUT_ROOT}/L128_k{k}_sigma{sigma}"
    os.makedirs(subdir, exist_ok=True)
    cache_dir = f"{subdir}/prop_cache"; os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'='*70}\n  k={k}  σ={sigma}  ({cfg['phase']})\n{'='*70}")

    # Train propagator
    print("loading train ...")
    train = load_train(k)
    # limit to 2048 for speed consistent with gen
    train_sub = train[..., :2048] if train.shape[-1] > 2048 else train
    print(f"  using {train_sub.shape[-1]} train configs")
    k_vals, Gt, Gte = propagator_cached(train_sub, "train", cache_dir)

    summary = dict()
    em_res = OrderedDict()
    ode_res = OrderedDict()

    for method, label in [("em", "SDE em"), ("ode", "ODE dpm2")]:
        print(f"\n-- method={method} --")
        for ep in EPOCHS:
            gen = load_gen(k, sigma, method, ep)
            if gen is None:
                print(f"   ep={ep}: missing, skip")
                continue
            k_v, G, Ge = propagator_cached(gen, f"{method}_ep{ep}", cache_dir)
            if method == "em":
                em_res[ep] = (G, Ge)
            else:
                ode_res[ep] = (G, Ge)
            print(f"   ep={ep}: computed ({gen.shape[-1]} configs)")

        # 4-panel at ep=10000
        if "10000" in (em_res if method == "em" else ode_res):
            res = (em_res if method == "em" else ode_res)["10000"]
            G_f, Ge_f = res
            stats = plot_4panel(k_vals, Gt, Gte, G_f, Ge_f,
                title=f"k={k}  σ={sigma}  {label}  ep=10000  (N=2048)",
                outpath=f"{subdir}/4panel_{method}_ep10000.png")
            summary[f"{method}_ep10000"] = stats
            print(f"   4-panel -> {subdir}/4panel_{method}_ep10000.png")
            print(f"     mean|Δr|={stats['mean_absratio']:.4f}  ΣKL={stats['sum_kl']:.4f}")

        # Evolution
        all_r = em_res if method == "em" else ode_res
        if len(all_r) > 1:
            plot_evolution(all_r, k_vals, Gt, k_label=str(k), method=method,
                outpath=f"{subdir}/evolution_{method}.png")
            print(f"   evolution -> {subdir}/evolution_{method}.png")

    # em vs ode (same IC) at ep=10000
    if "10000" in em_res and "10000" in ode_res:
        plot_em_vs_ode(k_vals, Gt, em_res, ode_res, k_label=str(k),
            outpath=f"{subdir}/em_vs_ode_ep10000.png")
        print(f"   em-vs-ode -> {subdir}/em_vs_ode_ep10000.png")

    return summary

if __name__ == "__main__":
    all_summary = {}
    for cfg in CFG:
        s = analyze_kappa(cfg)
        all_summary[cfg["k"]] = s

    print("\n" + "="*70)
    print("  SUMMARY  (ep=10000, N=2048)")
    print("="*70)
    print(f"{'k':>7}  {'method':>6}  {'mean|Δr|':>10}  {'max|Δr|':>10}  {'mean|z|':>10}  {'ΣKL':>8}")
    for k, s in all_summary.items():
        for key, st in s.items():
            method = key.split("_")[0]
            print(f"{k:>7.4f}  {method:>6s}  "
                  f"{st['mean_absratio']:>10.4f}  "
                  f"{st['max_absratio']:>10.4f}  "
                  f"{st['mean_absz']:>10.2f}  "
                  f"{st['sum_kl']:>8.4f}")
