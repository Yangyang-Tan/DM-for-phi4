"""Compare 3D em log-schedule vs linear-schedule at ep=10000."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from analyze_L64_3d_sweep import (build_bins_3d, compute_propagator_3d,
                                   per_mode_kl, zscore, phase_space_weighted,
                                   load_train)

ROOT = "/data/tyywork/DM/3Dphi4"
cache = f"{ROOT}/results/sigma_comparison_3D/L64_k0.2_sigma2760/prop_cache"

# Load cached train + em_log
d = np.load(f"{cache}/train.npz"); kv, Gt, Gte = d["k"], d["G"], d["Ge"]
d = np.load(f"{cache}/em_ep10000.npz"); G_log, Ge_log = d["G"], d["Ge"]

# Load linear sample and compute propagator
gen_lin = np.load(f"{ROOT}/runs/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/data"
                  "/samples_em_linear_steps2000_epoch=10000.npy").astype(np.float32)
print(f"linear gen shape={gen_lin.shape}  range=[{gen_lin.min():+.3f},{gen_lin.max():+.3f}]")

# moments
print("\n── moments ──")
gen_log = np.load(f"{ROOT}/runs/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/data"
                  "/samples_em_steps2000_epoch=10000.npy").astype(np.float32)
train = load_train()[..., :2048]

def moments(c, name):
    m = c.mean()
    per_c = c - c.mean(axis=(0,1,2), keepdims=True)
    v_c = (per_c**2).mean()
    rng = (c.min(), c.max())
    print(f"  {name:12s} ⟨φ⟩={m:+.4f}  ⟨φ²⟩_c={v_c:.4f}  range=[{rng[0]:+.3f},{rng[1]:+.3f}]")
    return v_c

v_train = moments(train, "train")
v_log   = moments(gen_log, "em log")
v_lin   = moments(gen_lin, "em linear")
print(f"\n  var_ratio log / train    = {v_log/v_train:.4f}")
print(f"  var_ratio linear / train = {v_lin/v_train:.4f}")

# compute linear propagator (cache if not already)
lin_cache = f"{cache}/em_linear_ep10000.npz"
if os.path.isfile(lin_cache):
    d = np.load(lin_cache); G_lin, Ge_lin = d["G"], d["Ge"]
    print("\n[cache] linear propagator loaded")
else:
    print("\ncomputing linear propagator ...")
    _, G_lin, Ge_lin = compute_propagator_3d(gen_lin, n_boot=200)
    np.savez(lin_cache, k=kv, G=G_lin, Ge=Ge_lin)

# diagnostics
nz = kv > 1e-8; k_nz = kv[nz]
Gt_nz, Gte_nz = Gt[nz], Gte[nz]

def diag(G, Ge, name):
    Gn, Gen = G[nz], Ge[nz]
    ratio = Gn / Gt_nz
    z = (Gn - Gt_nz) / np.sqrt(Gen**2 + Gte_nz**2 + 1e-30)
    kl = per_mode_kl(Gn, Gt_nz)
    wk = phase_space_weighted(k_nz, Gn, Gt_nz, D=3)
    print(f"  {name:12s} mean|Δr|={np.abs(ratio-1).mean():.4f}  "
          f"max|Δr|={np.abs(ratio-1).max():.4f}  "
          f"mean|z|={np.abs(z).mean():.2f}  "
          f"ΣKL={kl.sum():.4f}  "
          f"KL_IR={kl[k_nz<0.5].sum():.4f}  KL_UV={kl[k_nz>1.5].sum():.4f}")
    return ratio, z, kl, wk

print("\n── diagnostics (ep=10000, em) ──")
r_log, z_log, kl_log, w_log = diag(G_log, Ge_log, "log")
r_lin, z_lin, kl_lin, w_lin = diag(G_lin, Ge_lin, "linear")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
ax.errorbar(k_nz, Gt_nz, yerr=Gte_nz, fmt="o", ms=3, label="train", color="k", capsize=2)
ax.errorbar(k_nz, G_log[nz], yerr=Ge_log[nz], fmt="s", ms=3, label="em log", color="C0", capsize=2)
ax.errorbar(k_nz, G_lin[nz], yerr=Ge_lin[nz], fmt="^", ms=3, label="em linear", color="C3", capsize=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("|k|"); ax.set_ylabel("G(k)")
ax.set_title("3D k=0.2 ep=10000  em: log vs linear schedule")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
re_err_log = r_log * np.sqrt((Ge_log[nz]/G_log[nz])**2 + (Gte_nz/Gt_nz)**2)
re_err_lin = r_lin * np.sqrt((Ge_lin[nz]/G_lin[nz])**2 + (Gte_nz/Gt_nz)**2)
ax.errorbar(k_nz, r_log, yerr=re_err_log, fmt="s", ms=3, label="em log", color="C0", capsize=2)
ax.errorbar(k_nz, r_lin, yerr=re_err_lin, fmt="^", ms=3, label="em linear", color="C3", capsize=2)
ax.axhline(1.0, linestyle="--", color="gray")
ax.set_xscale("log"); ax.set_xlabel("|k|"); ax.set_ylabel("G_DM/G_train")
ax.set_title("ratio"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(k_nz, kl_log, "s", ms=3, label="em log", color="C0")
ax.plot(k_nz, kl_lin, "^", ms=3, label="em linear", color="C3")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("|k|"); ax.set_ylabel("D_k")
ax.set_title("per-mode Gaussian KL"); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
out = f"{ROOT}/results/sigma_comparison_3D/L64_k0.2_sigma2760/linear_vs_log_ep10000.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nsaved {out}")
