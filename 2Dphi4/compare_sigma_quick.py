"""
Quick σ-quality comparison based ONLY on existing data.

For k=0.28 (have both old σ=60 N=512 and new σ=640 N=2048):
  3-way comparison: train / old / new

For k=0.2705 (only have new σ=450 N=2048):
  2-way: train / new

Metrics:
  1) moments: ⟨φ⟩, ⟨φ²⟩_c  (centered variance)
  2) radial-averaged propagator G(|k|) via bootstrap
  3) propagator ratio G_gen / G_train
  4) integrated Gaussian KL  ΣD_k

Uses only numpy+h5py+scipy+matplotlib.
"""
import os, functools
from glob import glob
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

ROOT = "/data/tyywork/DM/2Dphi4"
OUT  = f"{ROOT}/sigma_comparison_L128"
os.makedirs(OUT, exist_ok=True)

# ---------------------------- loaders ---------------------------------
def load_train(k):
    path = f"{ROOT}/trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_128^2.jld2"
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    # normalize to (L, L, N) layout: h5py reads shape (N, L, L) from 2D jld2
    if cfgs.ndim == 3 and cfgs.shape[0] != cfgs.shape[1]:
        cfgs = cfgs.transpose(1, 2, 0)
    return cfgs  # (L, L, N)

def load_npy(path):
    return np.load(path).astype(np.float32)   # (L, L, N)

# ---------------------------- stats -----------------------------------
def moments(cfgs):
    """Return (global_mean, centered_variance) computed per-sample then averaged."""
    N = cfgs.shape[-1]
    per_sample_mean = cfgs.reshape(-1, N).mean(axis=0)         # (N,)
    per_sample_m2   = (cfgs.reshape(-1, N)**2).mean(axis=0)    # (N,)
    per_sample_var  = per_sample_m2 - per_sample_mean**2       # (N,)
    return per_sample_mean.mean(), per_sample_var.mean()

# ---------------------------- propagator -------------------------------
def compute_propagator(cfgs, max_diag=0.51, n_boot=200, seed=42):
    """Radially-averaged G(|k|) with bootstrap errors. cfgs: (L, L, N)."""
    L = cfgs.shape[0]
    N = cfgs.shape[-1]
    V = L * L
    # k-grid and diagonality filter
    ns = np.arange(L)
    ns = np.where(ns > L//2, ns - L, ns)
    KX, KY = np.meshgrid(ns, ns, indexing="ij")
    # lattice momentum squared
    kh2 = 4*np.sin(np.pi*KX/L)**2 + 4*np.sin(np.pi*KY/L)**2
    # per-mode p²
    p2x = 4*np.sin(np.pi*KX/L)**2
    p2y = 4*np.sin(np.pi*KY/L)**2
    sum_p2 = p2x + p2y
    sum_p4 = p2x**2 + p2y**2
    diag = np.zeros_like(sum_p2)
    mask0 = sum_p2 > 1e-10
    diag[mask0] = sum_p4[mask0] / sum_p2[mask0]**2
    # select modes with diagonality ≤ max_diag
    select = (diag <= max_diag) | ~mask0   # include k=0 where diag undefined
    ksq_unique = np.sort(np.unique(np.round(kh2[select], 5)))
    k_vals = np.sqrt(ksq_unique)
    nb = len(k_vals)
    # bin indices
    bin_indices = [np.where(np.abs(kh2 - ksq) < 1e-4) for ksq in ksq_unique]

    # per-config radial propagator
    G_rad = np.zeros((nb, N), dtype=np.float64)
    for i in range(N):
        cfg = cfgs[..., i].astype(np.float64)
        cfg = cfg - cfg.mean()
        phi_k = np.fft.fft2(cfg) / np.sqrt(V)
        ps = (phi_k * np.conj(phi_k)).real
        for j, idx in enumerate(bin_indices):
            G_rad[j, i] = ps[idx].mean()

    G_mean = G_rad.mean(axis=1)
    # bootstrap error
    rng = np.random.default_rng(seed)
    boots = np.zeros((n_boot, nb))
    for b in range(n_boot):
        ii = rng.integers(0, N, size=N)
        boots[b] = G_rad[:, ii].mean(axis=1)
    G_err = boots.std(axis=0)
    return k_vals, G_mean, G_err

def per_mode_kl(G_gen, G_train, eps=1e-30):
    Gg = np.maximum(G_gen,   eps)
    Gt = np.maximum(G_train, eps)
    r = Gt / Gg
    return 0.5 * (r - 1 - np.log(r))

# ---------------------------- driver ----------------------------------
def analyze(k, new_sigma, old_sigma=None, cache={}):
    print(f"\n{'='*60}\n  k = {k}   new σ = {new_sigma}"
          f"{'' if old_sigma is None else f'   old σ = {old_sigma}'}\n{'='*60}")

    # ---- load ----
    print("loading train ...", flush=True)
    if f"train_{k}" not in cache:
        cache[f"train_{k}"] = load_train(k)
    train = cache[f"train_{k}"]
    print(f"  train shape = {train.shape}")

    new_path = (f"{ROOT}/phi4_L128_k{k}_l0.022_ncsnpp_sigma{new_sigma}"
                f"/data/samples_em_steps2000_epoch=10000.npy")
    new = load_npy(new_path)
    print(f"  new   shape = {new.shape}    range=[{new.min():+.3f},{new.max():+.3f}]")

    old = None
    if old_sigma is not None:
        old_path = (f"{ROOT}/phi4_L128_k{k}_l0.022_ncsnpp"
                    f"/data/samples_em_steps2000_epoch=9999.npy")
        if os.path.isfile(old_path):
            old = load_npy(old_path)
            print(f"  old   shape = {old.shape}    range=[{old.min():+.3f},{old.max():+.3f}]")
        else:
            print(f"  [skip] no old file at {old_path}")

    # ---- moments ----
    m_t, var_t = moments(train)
    m_n, var_n = moments(new)
    print("\n── moments (per-config mean, per-config centered ⟨φ²⟩) ──")
    print(f"  train  : ⟨φ⟩={m_t:+.4f}   ⟨φ²⟩_c={var_t:.4f}")
    print(f"  new σ={new_sigma}: ⟨φ⟩={m_n:+.4f}   ⟨φ²⟩_c={var_n:.4f}   "
          f"ratio(var) = {var_n/var_t:.4f}")
    if old is not None:
        m_o, var_o = moments(old)
        print(f"  old σ={old_sigma:3d}: ⟨φ⟩={m_o:+.4f}   ⟨φ²⟩_c={var_o:.4f}   "
              f"ratio(var) = {var_o/var_t:.4f}")

    # ---- propagators ----
    print("\ncomputing propagator (train) ...", flush=True)
    kv, Gt, Gte = compute_propagator(train[..., :2048] if train.shape[-1] > 2048
                                     else train, n_boot=200)
    # use at most 2048 train configs for speed (10240 → 2048 is fine for propagator)
    print("computing propagator (new) ...", flush=True)
    _, Gn, Gne = compute_propagator(new, n_boot=200)
    if old is not None:
        print("computing propagator (old) ...", flush=True)
        _, Go, Goe = compute_propagator(old, n_boot=200)
    else:
        Go = Goe = None

    nz = kv > 1e-8
    kvnz = kv[nz]
    Gt_n, Gte_n = Gt[nz], Gte[nz]
    Gn_n, Gne_n = Gn[nz], Gne[nz]
    ratio_new = Gn_n / Gt_n
    ratio_new_err = ratio_new * np.sqrt((Gne_n/Gn_n)**2 + (Gte_n/Gt_n)**2)
    kl_new = per_mode_kl(Gn_n, Gt_n)

    print("\n── propagator summary (nonzero k bins) ──")
    print(f"  new σ={new_sigma}: max|r-1|={np.abs(ratio_new-1).max():.4f}   "
          f"mean|r-1|={np.abs(ratio_new-1).mean():.4f}   ΣD_k={kl_new.sum():.4f}")
    if Go is not None:
        Go_n, Goe_n = Go[nz], Goe[nz]
        ratio_old = Go_n / Gt_n
        ratio_old_err = ratio_old * np.sqrt((Goe_n/Go_n)**2 + (Gte_n/Gt_n)**2)
        kl_old = per_mode_kl(Go_n, Gt_n)
        print(f"  old σ={old_sigma}: max|r-1|={np.abs(ratio_old-1).max():.4f}   "
              f"mean|r-1|={np.abs(ratio_old-1).mean():.4f}   ΣD_k={kl_old.sum():.4f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.errorbar(kvnz, Gt_n, yerr=Gte_n, fmt="o", ms=3, label="train",
                color="black", capsize=2)
    ax.errorbar(kvnz, Gn_n, yerr=Gne_n, fmt="s", ms=3,
                label=f"new σ={new_sigma}", color="steelblue", capsize=2)
    if Go is not None:
        ax.errorbar(kvnz, Go_n, yerr=Goe_n, fmt="^", ms=3,
                    label=f"old σ={old_sigma}", color="orange", capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("G(k)")
    ax.set_title(f"propagator  (k={k})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(kvnz, ratio_new, yerr=ratio_new_err, fmt="s", ms=3,
                label=f"new σ={new_sigma}", color="steelblue", capsize=2)
    if Go is not None:
        ax.errorbar(kvnz, ratio_old, yerr=ratio_old_err, fmt="^", ms=3,
                    label=f"old σ={old_sigma}", color="orange", capsize=2)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("G_gen / G_train")
    ax.set_title(f"ratio (k={k})"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(kvnz, kl_new, "s", ms=3, label=f"new σ={new_sigma}", color="steelblue")
    if Go is not None:
        ax.plot(kvnz, kl_old, "^", ms=3, label=f"old σ={old_sigma}", color="orange")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|k|"); ax.set_ylabel("per-mode KL D_k")
    ax.set_title(f"Gaussian KL (k={k})"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    outfile = f"{OUT}/sigma_quick_k{k}.png"
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.savefig(outfile.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  ✓ saved {outfile}")

    out = {"k": k, "new_sigma": new_sigma,
           "var_ratio_new": var_n/var_t,
           "mean_absratio_new": float(np.abs(ratio_new-1).mean()),
           "KL_new": float(kl_new.sum()),
           "mean_phi_new_minus_train": float(m_n - m_t)}
    if Go is not None:
        out["old_sigma"] = old_sigma
        out["var_ratio_old"] = var_o/var_t
        out["mean_absratio_old"] = float(np.abs(ratio_old-1).mean())
        out["KL_old"] = float(kl_old.sum())
        out["mean_phi_old_minus_train"] = float(m_o - m_t)
    return out

if __name__ == "__main__":
    cache = {}
    results = []
    # k=0.2705: new only (2-way)
    results.append(analyze(0.2705, new_sigma=450, old_sigma=None, cache=cache))
    # k=0.28: 3-way
    results.append(analyze(0.28,   new_sigma=640, old_sigma=60,  cache=cache))

    print("\n" + "="*68)
    print("  SUMMARY")
    print("="*68)
    print(f"{'k':>7}  {'σ':>8}  {'ratio(var)':>10}  {'mean|ΔG/G|':>10}  {'ΣD_k':>8}")
    for r in results:
        if "old_sigma" in r:
            print(f"{r['k']:>7.4f}  old={r['old_sigma']:>4d}  "
                  f"{r['var_ratio_old']:>10.4f}  {r['mean_absratio_old']:>10.4f}  "
                  f"{r['KL_old']:>8.4f}")
        print(f"{r['k']:>7.4f}  new={r['new_sigma']:>4d}  "
              f"{r['var_ratio_new']:>10.4f}  {r['mean_absratio_new']:>10.4f}  "
              f"{r['KL_new']:>8.4f}")
