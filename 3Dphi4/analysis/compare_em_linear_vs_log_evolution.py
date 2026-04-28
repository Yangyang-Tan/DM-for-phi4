"""
3D em linear-schedule vs log-schedule evolution comparison (full 18 epochs).

- Loads LINEAR propagator cache (already present in prop_cache/) as current
- Recomputes LOG propagators from data_log/ (old samples preserved there)
- Stores log caches in prop_cache_log/ for reuse
- Produces 4-panel comparison figure:
    (a) ΣKL vs epoch, both schedules
    (b) var_ratio vs epoch
    (c) mean|Δr| vs epoch
    (d) ratio curves at ep=10000 (final), both schedules

Expected story: linear almost always << log, especially at late epochs.
"""
import os, glob, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from analyze_L64_3d_sweep import compute_propagator_3d, per_mode_kl

ROOT = "/data/tyywork/DM/3Dphi4"
CACHE_LIN = f"{ROOT}/sigma_comparison_3D/L64_k0.2_sigma2760/prop_cache"
CACHE_LOG = f"{ROOT}/sigma_comparison_3D/L64_k0.2_sigma2760/prop_cache_log"
DATA_LIN  = f"{ROOT}/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/data"
DATA_LOG  = f"{ROOT}/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/data_log"
os.makedirs(CACHE_LOG, exist_ok=True)

EPOCHS = ["0001","0002","0003","0005","0009","0016","0028","0045","0079",
          "0138","0242","0422","0739","1291","2257","3593","6280","10000"]

def get_prop_log(ep):
    p = f"{CACHE_LOG}/em_ep{ep}.npz"
    if os.path.isfile(p):
        d = np.load(p); return d["k"], d["G"], d["Ge"]
    src = f"{DATA_LOG}/samples_em_steps2000_epoch={ep}.npy"
    if not os.path.isfile(src):
        return None
    cfgs = np.load(src).astype(np.float32)
    kv, G, Ge = compute_propagator_3d(cfgs, n_boot=200)
    np.savez(p, k=kv, G=G, Ge=Ge)
    return kv, G, Ge

def get_prop_lin(ep):
    p = f"{CACHE_LIN}/em_ep{ep}.npz"
    if not os.path.isfile(p): return None
    d = np.load(p); return d["k"], d["G"], d["Ge"]

def moments_from_samples(src):
    """Per-config centered <phi^2>."""
    if not os.path.isfile(src): return None
    c = np.load(src).astype(np.float32)
    per_c = c - c.mean(axis=(0, 1, 2), keepdims=True)
    return (per_c**2).mean()

# Train reference
d = np.load(f"{CACHE_LIN}/train.npz"); kv_t, Gt, Gte = d["k"], d["G"], d["Ge"]
nz = kv_t > 1e-8
k_nz = kv_t[nz]; Gt_nz, Gte_nz = Gt[nz], Gte[nz]

# Train variance reference
import h5py
with h5py.File(f"{ROOT}/trainingdata/cfgs_wolff_fahmc_k=0.2_l=0.9_64^3.jld2", "r") as f:
    train_c = np.array(f["cfgs"][:2048], dtype=np.float32)
train_perc = train_c - train_c.mean(axis=(1, 2, 3), keepdims=True)
v_train = (train_perc**2).mean()
print(f"train variance (per-config centered) = {v_train:.4f}")
del train_c, train_perc

# Collect metrics
results = {"log": {}, "lin": {}}
for schedule, get_fn, src_dir in [("log", get_prop_log, DATA_LOG),
                                  ("lin", get_prop_lin, DATA_LIN)]:
    print(f"\n-- {schedule} schedule --")
    for ep in EPOCHS:
        out = get_fn(ep)
        if out is None:
            print(f"  ep={ep}: missing"); continue
        _, G, Ge = out
        Gn = G[nz]
        ratio = Gn / Gt_nz
        kl = per_mode_kl(Gn, Gt_nz)
        # var from samples if needed
        src = f"{src_dir}/samples_em_steps2000_epoch={ep}.npy"
        vg = moments_from_samples(src)
        results[schedule][ep] = dict(
            ratio=ratio, kl=kl, sum_kl=float(kl.sum()),
            mean_absratio=float(np.abs(ratio-1).mean()),
            max_absratio=float(np.abs(ratio-1).max()),
            var_ratio=(vg / v_train) if vg is not None else np.nan,
        )
        print(f"  ep={ep}  ΣKL={results[schedule][ep]['sum_kl']:.4f}  "
              f"mean|Δr|={results[schedule][ep]['mean_absratio']:.4f}  "
              f"var_ratio={results[schedule][ep]['var_ratio']:.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
eps_arr = np.array([int(e) for e in EPOCHS])

# (a) ΣKL vs epoch
ax = axes[0, 0]
for sched, color, marker in [("log", "C3", "s"), ("lin", "C0", "o")]:
    ys = [results[sched][e]["sum_kl"] for e in EPOCHS if e in results[sched]]
    xs = [int(e) for e in EPOCHS if e in results[sched]]
    ax.plot(xs, ys, f"{marker}-", color=color, ms=5,
            label=f"{'log' if sched=='log' else 'linear'} schedule")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("Σ D_k (integrated KL)")
ax.set_title("(a)  ΣKL vs epoch"); ax.legend(); ax.grid(alpha=0.3)

# (b) var_ratio vs epoch
ax = axes[0, 1]
for sched, color, marker in [("log", "C3", "s"), ("lin", "C0", "o")]:
    ys = [results[sched][e]["var_ratio"] for e in EPOCHS if e in results[sched]]
    xs = [int(e) for e in EPOCHS if e in results[sched]]
    ax.plot(xs, ys, f"{marker}-", color=color, ms=5,
            label=f"{'log' if sched=='log' else 'linear'}")
ax.axhline(1.0, linestyle="--", color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("⟨φ²⟩_gen / ⟨φ²⟩_train")
ax.set_title("(b)  variance ratio vs epoch"); ax.legend(); ax.grid(alpha=0.3)

# (c) mean|Δr| vs epoch
ax = axes[1, 0]
for sched, color, marker in [("log", "C3", "s"), ("lin", "C0", "o")]:
    ys = [results[sched][e]["mean_absratio"] for e in EPOCHS if e in results[sched]]
    xs = [int(e) for e in EPOCHS if e in results[sched]]
    ax.plot(xs, ys, f"{marker}-", color=color, ms=5,
            label=f"{'log' if sched=='log' else 'linear'}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("mean |G_DM/G_train − 1|")
ax.set_title("(c)  mean propagator ratio deviation"); ax.legend(); ax.grid(alpha=0.3)

# (d) ratio curves at ep=10000
ax = axes[1, 1]
r_log = results["log"].get("10000", None)
r_lin = results["lin"].get("10000", None)
if r_log is not None:
    ax.plot(k_nz, r_log["ratio"], "s", ms=3, color="C3",
            label=f"log  (ΣKL={r_log['sum_kl']:.2f})")
if r_lin is not None:
    ax.plot(k_nz, r_lin["ratio"], "o", ms=3, color="C0",
            label=f"linear  (ΣKL={r_lin['sum_kl']:.2f})")
ax.axhline(1.0, linestyle="--", color="gray")
ax.set_xscale("log")
ax.set_xlabel("|k|"); ax.set_ylabel("G_DM / G_train")
ax.set_title("(d)  ratio vs |k| at ep=10000")
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle(
    "3D L=64 k=0.2 l=0.9 σ=2760  —  em (SDE) schedule comparison (full 18 epochs)",
    fontsize=13)
plt.tight_layout()
out = f"{ROOT}/sigma_comparison_3D/L64_k0.2_sigma2760/em_linear_vs_log_evolution.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nsaved {out}")

# Print summary table
print("\n" + "="*70)
print("  SUMMARY: log vs linear schedule improvement (em, ep=10000)")
print("="*70)
log10 = results["log"].get("10000"); lin10 = results["lin"].get("10000")
if log10 and lin10:
    print(f"{'metric':25s}  {'log':>10s}  {'linear':>10s}  {'improvement':>12s}")
    for key, nice in [("sum_kl", "ΣKL"),
                      ("mean_absratio", "mean|Δr|"),
                      ("max_absratio", "max|Δr|"),
                      ("var_ratio", "var_ratio")]:
        lo, li = log10[key], lin10[key]
        if key == "var_ratio":
            imp = f"{abs(lo-1)/max(abs(li-1),1e-10):.1f}× closer to 1"
        else:
            imp = f"{lo/max(li,1e-10):.1f}×"
        print(f"{nice:25s}  {lo:>10.4f}  {li:>10.4f}  {imp:>12s}")
