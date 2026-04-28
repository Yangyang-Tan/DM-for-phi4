"""
3D L=64 phi^4 sweep analysis — mirror of 2D's analyze_L128_sweep.py +
analyze_L128_diagnostics_evolution.py, but with 3D FFT and D=3 phase-space weight.

For k=0.2, σ=2760, method ∈ {em, ode}:
  - Load 18 log-spaced epochs of 256 samples each
  - Compute 3D radial-averaged G(|k|) vs training-data propagator
  - Produce CelebA-style 4-panel at ep=10000
  - Produce multi-epoch evolution
  - Produce em-vs-ode comparison at ep=10000
  - Produce UV/IR diagnostics evolution (KL, z, phase-space weighted)

Outputs under sigma_comparison_3D/L64_k0.2_sigma2760/.
"""
import os, time
from collections import OrderedDict
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = "/data/tyywork/DM/3Dphi4"
OUT_ROOT = f"{ROOT}/sigma_comparison_3D"
os.makedirs(OUT_ROOT, exist_ok=True)

K = 0.2
L = 64
SIGMA = 2760
EPOCHS = ["0001","0002","0003","0005","0009","0016","0028","0045","0079",
          "0138","0242","0422","0739","1291","2257","3593","6280","10000"]
METHODS = ["em", "ode"]

# ---------------------------- loaders ---------------------------------
def load_train():
    path = f"{ROOT}/trainingdata/cfgs_wolff_fahmc_k={K}_l=0.9_64^3.jld2"
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    # expected layout (N, L, L, L) -> (L, L, L, N)
    if cfgs.shape[0] != L and cfgs.shape[-1] != L:
        raise ValueError(f"unexpected shape {cfgs.shape}")
    if cfgs.shape[0] != L:  # (N, L, L, L)
        cfgs = cfgs.transpose(1, 2, 3, 0)
    return cfgs

def load_gen(method, ep):
    steps = 2000 if method == "em" else 400
    path = (f"{ROOT}/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma{SIGMA}"
            f"/data/samples_{method}_steps{steps}_epoch={ep}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path).astype(np.float32)  # (L, L, L, N)

# ---------------------------- 3D propagator ---------------------------
def build_bins_3d(L=64, max_diag=0.51):
    ns = np.arange(L); ns = np.where(ns > L//2, ns - L, ns)
    KX, KY, KZ = np.meshgrid(ns, ns, ns, indexing="ij")
    # lattice k-squared
    kh2 = 4*np.sin(np.pi*KX/L)**2 + 4*np.sin(np.pi*KY/L)**2 + 4*np.sin(np.pi*KZ/L)**2
    p2x = 4*np.sin(np.pi*KX/L)**2
    p2y = 4*np.sin(np.pi*KY/L)**2
    p2z = 4*np.sin(np.pi*KZ/L)**2
    sum_p2 = p2x + p2y + p2z
    sum_p4 = p2x**2 + p2y**2 + p2z**2
    diag = np.zeros_like(sum_p2); m0 = sum_p2 > 1e-10
    diag[m0] = sum_p4[m0] / sum_p2[m0]**2
    sel = (diag <= max_diag) | ~m0
    ksq_u = np.sort(np.unique(np.round(kh2[sel], 5)))
    k_vals = np.sqrt(ksq_u)
    bin_idx = [np.where(np.abs(kh2 - ks) < 1e-4) for ks in ksq_u]
    return k_vals, bin_idx

def compute_propagator_3d(cfgs, bins=None, n_boot=200, seed=42):
    Lcfg = cfgs.shape[0]; N = cfgs.shape[-1]; V = Lcfg**3
    if bins is None:
        k_vals, bin_idx = build_bins_3d(Lcfg)
    else:
        k_vals, bin_idx = bins
    nb = len(k_vals)
    G_rad = np.zeros((nb, N), dtype=np.float64)
    for i in range(N):
        cfg = cfgs[..., i].astype(np.float64)
        cfg = cfg - cfg.mean()
        ps = np.abs(np.fft.fftn(cfg))**2 / V
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

def zscore(G_gen, Ge_gen, G_train, Ge_train, eps=1e-30):
    denom = np.maximum(np.sqrt(Ge_gen**2 + Ge_train**2), eps)
    return (G_gen - G_train) / denom

def phase_space_weighted(k_vals, G_gen, G_train, D=3):
    return (k_vals ** (D - 1)) * (G_gen - G_train)  # k² for D=3

# ---------------------------- caching ----------------------------------
def propagator_cached(cfgs, label, cache_dir):
    cache = f"{cache_dir}/{label}.npz"
    if os.path.isfile(cache):
        d = np.load(cache)
        return d["k"], d["G"], d["Ge"]
    t0 = time.time()
    k_vals, G, Ge = compute_propagator_3d(cfgs)
    print(f"      computed in {time.time()-t0:.1f}s (N={cfgs.shape[-1]})")
    np.savez(cache, k=k_vals, G=G, Ge=Ge)
    return k_vals, G, Ge

# ---------------------------- plots -----------------------------------
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
    ax.set_xlabel("|k|"); ax.set_ylabel("G(k)"); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.errorbar(kv, ratio, yerr=ratio_err, fmt="s", ms=3, color="C0", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("G_DM/G_train")
    ax.set_title("ratio"); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(kv, z, "s", ms=3, color="C0")
    for y in (-2, 0, 2): ax.axhline(y, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("z_k")
    ax.set_title("z-score"); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(kv, kl, "s", ms=3, color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("D_k")
    ax.set_title("Gaussian KL"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)
    return dict(
        max_absratio=float(np.abs(ratio-1).max()),
        mean_absratio=float(np.abs(ratio-1).mean()),
        mean_absz=float(np.abs(z).mean()),
        max_absz=float(np.abs(z).max()),
        sum_kl=float(kl.sum()),
    )

def plot_evolution(all_res, k_vals, Gt, method, outpath):
    if not all_res: return
    nb = len(k_vals)
    nz_first = np.where(k_vals > 1e-8)[0][0]
    k_highlights = np.unique([nz_first, nz_first+3, nb//4, nb//2, nb*3//4, nb-2])
    eps_sorted = sorted(all_res, key=int)
    ep_nums = np.array([int(e) for e in eps_sorted])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    cmap = mpl.colormaps["viridis"]
    for i, ki in enumerate(k_highlights):
        if k_vals[ki] < 1e-8: continue
        ratios = [all_res[ep][0][ki]/Gt[ki] for ep in eps_sorted]
        errs   = [all_res[ep][1][ki]/Gt[ki] for ep in eps_sorted]
        c = cmap(i / max(1, len(k_highlights)-1))
        ax.errorbar(ep_nums, ratios, yerr=errs, fmt="o-", ms=4, color=c,
                    label=f"|k|={k_vals[ki]:.3f}", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("G_DM/G_train")
    ax.set_title(f"evolution (3D, k=0.2, {method})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    cmap2 = mpl.colormaps["plasma"]
    for j, ep in enumerate(eps_sorted):
        Gg, Gge = all_res[ep]; nzm = k_vals > 1e-8
        ratio = Gg[nzm]/Gt[nzm]
        c = cmap2(j / max(1, len(eps_sorted)-1))
        ax.plot(k_vals[nzm], ratio, "-", color=c, alpha=0.7)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("G_DM/G_train")
    ax.set_title("ratio across epochs (purple→yellow)")
    sm = mpl.cm.ScalarMappable(cmap=cmap2,
        norm=mpl.colors.LogNorm(vmin=ep_nums.min()+1, vmax=ep_nums.max()))
    plt.colorbar(sm, ax=ax).set_label("epoch")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_em_vs_ode(k_vals, Gt, em_res, ode_res, outpath):
    ep = "10000"
    if ep not in em_res or ep not in ode_res: return
    Gem, Gem_e = em_res[ep]; Gode, Gode_e = ode_res[ep]
    nz = k_vals > 1e-8
    kv = k_vals[nz]
    ratio_em = Gem[nz]/Gt[nz]; ratio_ode = Gode[nz]/Gt[nz]
    re_err = ratio_em * (Gem_e[nz]/Gem[nz])
    ro_err = ratio_ode * (Gode_e[nz]/Gode[nz])
    em_ode_diff = (Gem[nz] - Gode[nz]) / ((Gem[nz] + Gode[nz]) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.errorbar(kv, ratio_em, yerr=re_err, fmt="s", ms=3, label="em (SDE)",
                color="C0", capsize=2)
    ax.errorbar(kv, ratio_ode, yerr=ro_err, fmt="^", ms=3, label="ode (DPM-2)",
                color="C3", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("G_DM/G_train")
    ax.set_title(f"em vs ode, ep=10000 (3D k=0.2)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(kv, em_ode_diff*100, "o", ms=3, color="C4")
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|")
    ax.set_ylabel("(G_em - G_ode) / avg × 100%")
    ax.set_title("em - ode (same IC)"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_diagnostics_evolution(method, results, k_vals, Gt, Gte, outpath):
    nz = k_vals > 1e-8
    k_nz = k_vals[nz]
    Gt_nz, Gte_nz = Gt[nz], Gte[nz]
    nb = len(k_nz)
    # Precompute diagnostics
    for ep in results:
        Gg, Gge = results[ep]
        results[ep] = {
            "Gg": Gg[nz], "Gge": Gge[nz],
            "Dk": per_mode_kl(Gg[nz], Gt_nz),
            "zk": zscore(Gg[nz], Gge[nz], Gt_nz, Gte_nz),
            "wk": phase_space_weighted(k_nz, Gg[nz], Gt_nz, D=3),
        }
    eps_sorted = sorted(results, key=int)
    ep_nums = np.array([int(e) for e in eps_sorted])
    idx_rep = np.unique([0, nb // 6, nb // 3, nb // 2, 2*nb // 3, nb - 2])
    ep_colors = mpl.colormaps["plasma"](
        (np.log(ep_nums) - np.log(ep_nums.min())) /
        (np.log(ep_nums.max()) - np.log(ep_nums.min())))
    kbin_colors = mpl.colormaps["viridis"](np.linspace(0, 1, len(idx_rep)))
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    # Row 0: KL
    ax = axes[0,0]
    for ci, ki in enumerate(idx_rep):
        ys = [results[ep]["Dk"][ki] for ep in eps_sorted]
        ax.plot(ep_nums, ys, "o-", ms=4, color=kbin_colors[ci], label=f"|k|={k_nz[ki]:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("D_k")
    ax.set_title("KL vs epoch"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax = axes[0,1]
    for j, ep in enumerate(eps_sorted):
        ax.plot(k_nz, results[ep]["Dk"], "-", color=ep_colors[j], alpha=0.8, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("D_k")
    ax.set_title("KL vs |k|"); ax.grid(alpha=0.3)
    # Row 1: |z|
    ax = axes[1,0]
    for ci, ki in enumerate(idx_rep):
        ys = [abs(results[ep]["zk"][ki]) for ep in eps_sorted]
        ax.plot(ep_nums, ys, "o-", ms=4, color=kbin_colors[ci], label=f"|k|={k_nz[ki]:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("|z_k|")
    ax.set_title("|z-score| vs epoch"); ax.axhline(2.0, linestyle="--", color="gray", lw=0.8)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax = axes[1,1]
    for j, ep in enumerate(eps_sorted):
        ax.plot(k_nz, results[ep]["zk"], "-", color=ep_colors[j], alpha=0.8, lw=1)
    ax.axhline(0.0, linestyle="--", color="gray")
    for y in (-2, 2): ax.axhline(y, linestyle=":", color="gray", lw=0.7)
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("z_k")
    ax.set_title("z-score vs |k|"); ax.grid(alpha=0.3)
    # Row 2: w_k
    ax = axes[2,0]
    for ci, ki in enumerate(idx_rep):
        ys = [results[ep]["wk"][ki] for ep in eps_sorted]
        ax.plot(ep_nums, ys, "o-", ms=4, color=kbin_colors[ci], label=f"|k|={k_nz[ki]:.3f}")
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("w_k = k² · ΔG  (D=3)")
    ax.set_title("phase-space weighted Δ vs epoch")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax = axes[2,1]
    for j, ep in enumerate(eps_sorted):
        ax.plot(k_nz, results[ep]["wk"], "-", color=ep_colors[j], alpha=0.8, lw=1)
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("w_k")
    ax.set_title("w_k vs |k|"); ax.grid(alpha=0.3)
    sm = mpl.cm.ScalarMappable(cmap=mpl.colormaps["plasma"],
        norm=mpl.colors.LogNorm(vmin=ep_nums.min()+1, vmax=ep_nums.max()))
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(sm, cax=cbar_ax).set_label("epoch")
    plt.suptitle(f"3D L=64 k={K} σ={SIGMA}  method={method}  UV/IR diagnostics vs epoch",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.savefig(outpath.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)
    # Integrated metrics per epoch
    out = OrderedDict()
    for ep in eps_sorted:
        out[ep] = dict(
            KL_sum=float(results[ep]["Dk"].sum()),
            z_rmse=float(np.sqrt(np.mean(results[ep]["zk"]**2))),
            KL_IR=float(results[ep]["Dk"][k_nz < 0.5].sum()),
            KL_UV=float(results[ep]["Dk"][k_nz > 1.5].sum()),
            w_abs_IR=float(np.abs(results[ep]["wk"][k_nz < 0.5]).mean()),
            w_abs_UV=float(np.abs(results[ep]["wk"][k_nz > 1.5]).mean()),
        )
    return out

# ---------------------------- main -----------------------------------
def main():
    subdir = f"{OUT_ROOT}/L64_k{K}_sigma{SIGMA}"
    os.makedirs(subdir, exist_ok=True)
    cache = f"{subdir}/prop_cache"; os.makedirs(cache, exist_ok=True)

    print(f"\n{'='*60}\n  3D L=64  k={K}  σ={SIGMA}\n{'='*60}")

    print("loading train (this can take a minute for 5 GB)...")
    train = load_train()
    # limit to 2048 configs for speed; train has 2560
    train_sub = train[..., :2048] if train.shape[-1] > 2048 else train
    print(f"  train shape = {train_sub.shape}")
    print("computing train propagator ...")
    k_vals, Gt, Gte = propagator_cached(train_sub, "train", cache)

    # quick per-config moments (train)
    m = float(train.mean())
    v = float(((train - train.mean(axis=(0,1,2), keepdims=True))**2).mean())
    print(f"  train  ⟨φ⟩={m:+.4f}  ⟨φ²⟩_c={v:.4f}")
    del train, train_sub

    summary = {}
    all_res = {}
    for method in METHODS:
        print(f"\n-- method={method} --")
        res = OrderedDict()
        for ep in EPOCHS:
            label = f"{method}_ep{ep}"
            cache_file = f"{cache}/{label}.npz"
            if os.path.isfile(cache_file):
                k_, G_, Ge_ = propagator_cached(None, label, cache)
                res[ep] = (G_, Ge_)
                print(f"   ep={ep}: cached")
                continue
            gen = load_gen(method, ep)
            if gen is None:
                print(f"   ep={ep}: missing, skip"); continue
            k_, G_, Ge_ = propagator_cached(gen, label, cache)
            res[ep] = (G_, Ge_)
            # per-config mean, var
            m_g = float(gen.mean())
            v_g = float(((gen - gen.mean(axis=(0,1,2), keepdims=True))**2).mean())
            print(f"   ep={ep}: ⟨φ⟩={m_g:+.4f}  ⟨φ²⟩_c={v_g:.4f}  var_ratio={v_g/v:.4f}")
            del gen
        if "10000" in res:
            G_f, Ge_f = res["10000"]
            stats = plot_4panel(k_vals, Gt, Gte, G_f, Ge_f,
                title=f"3D k={K} σ={SIGMA} {method.upper()} ep=10000 (N=256)",
                outpath=f"{subdir}/4panel_{method}_ep10000.png")
            summary[method] = stats
            print(f"   4-panel -> {subdir}/4panel_{method}_ep10000.png")
            print(f"     mean|Δr|={stats['mean_absratio']:.4f}  ΣKL={stats['sum_kl']:.4f}")
        if len(res) > 1:
            plot_evolution(res, k_vals, Gt, method,
                f"{subdir}/evolution_{method}.png")
            print(f"   evolution -> {subdir}/evolution_{method}.png")
        # Diagnostic evolution (make a copy of res because we mutate it)
        res_copy = OrderedDict((k_, v_) for k_, v_ in res.items())
        tot = plot_diagnostics_evolution(method, res_copy, k_vals, Gt, Gte,
            f"{subdir}/diagnostics_evolution_{method}.png")
        all_res[method] = (res, tot)
        print(f"   diagnostics -> {subdir}/diagnostics_evolution_{method}.png")

    # em vs ode
    if "em" in all_res and "ode" in all_res:
        plot_em_vs_ode(k_vals, Gt, all_res["em"][0], all_res["ode"][0],
            f"{subdir}/em_vs_ode_ep10000.png")
        print(f"\nem-vs-ode -> {subdir}/em_vs_ode_ep10000.png")

    # Integrated summary
    print("\n" + "="*70)
    print("  SUMMARY  (ep=10000, N=256)")
    print("="*70)
    print(f"{'method':>6}  {'mean|Δr|':>10}  {'max|Δr|':>10}  {'mean|z|':>8}  {'ΣKL':>8}  "
          f"{'KL_IR':>8}  {'KL_UV':>8}  {'|w|_IR':>9}  {'|w|_UV':>9}")
    for method in METHODS:
        if method not in summary: continue
        s = summary[method]
        tot = all_res[method][1]["10000"]
        print(f"{method:>6}  {s['mean_absratio']:>10.4f}  {s['max_absratio']:>10.4f}  "
              f"{s['mean_absz']:>8.2f}  {s['sum_kl']:>8.4f}  "
              f"{tot['KL_IR']:>8.4f}  {tot['KL_UV']:>8.4f}  "
              f"{tot['w_abs_IR']:>9.4f}  {tot['w_abs_UV']:>9.4f}")


if __name__ == "__main__":
    main()
