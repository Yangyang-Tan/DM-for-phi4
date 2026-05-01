"""
Analyze MALA acceptance rate vs training epoch for different (L, κ) configs.

Uses fixed training-data configurations as the starting point so that
acceptance rate purely reflects score quality (not sample location).

Usage:
    python analyze_accept_rate.py --device cuda:0
    python analyze_accept_rate.py --device cuda:0 --L 8 16 32 --k 0.26 0.2705
    python analyze_accept_rate.py --plot_only          # re-plot from existing CSV
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import csv
import re
import functools
import argparse
from pathlib import Path

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from sample_phi4 import phi4_action


# ---------------------------------------------------------------------------
#  Discovery helpers
# ---------------------------------------------------------------------------

def discover_configs(base_dir, network="ncsnpp"):
    """Return sorted list of (L, k, l, model_dir) from directory names."""
    configs = []
    for d in sorted(Path(base_dir).iterdir()):
        m = re.match(
            rf'phi4_L(\d+)_k([\d.]+)_l([\d.]+)_{re.escape(network)}$', d.name
        )
        if m and d.is_dir():
            L, k, l = int(m.group(1)), float(m.group(2)), float(m.group(3))
            configs.append((L, k, l, d))
    configs.sort(key=lambda x: (x[0], x[1]))
    return configs


def find_epoch_checkpoints(model_dir):
    """Map epoch → checkpoint path, skipping -v* duplicates."""
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=epoch=*.ckpt")):
        if '-v' in f.stem:
            continue
        m = re.search(r'epoch=(\d+)', f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


# ---------------------------------------------------------------------------
#  Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    """Load training data, normalise to [−1, 1], return (num_samples, 1, L, L)."""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.default_rng(42).choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def run_mala_diagnostic(ckpt_path, device, x_ref, mh_steps=50,
                        t_mh=0.01, mala_step_size=None):
    """Load one checkpoint, run score_diagnostic on fixed reference data."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    del ckpt

    path_str = str(ckpt_path)
    k = float(re.search(r'_k([\d.]+)', path_str).group(1))
    l = float(re.search(r'_l([\d.]+)', path_str).group(1))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)
    model = DiffusionModel.load_from_checkpoint(
        str(ckpt_path), score_model=score_model
    )
    model = model.to(device).eval()

    action_fn = functools.partial(
        phi4_action, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )

    _, accept_rate = model.score_diagnostic(
        x_ref, action_fn,
        t_mh=t_mh, mh_steps=mh_steps,
        mala_step_size=mala_step_size,
    )

    mean_acc = accept_rate.mean().item()
    std_acc = accept_rate.std().item()

    del model, score_model
    torch.cuda.empty_cache()
    return mean_acc, std_acc


# ---------------------------------------------------------------------------
#  CSV I/O (for resume support)
# ---------------------------------------------------------------------------

FIELDS = ['L', 'k', 'l', 'epoch', 'accept_mean', 'accept_std']


def load_csv(path):
    """Return (list-of-dicts, set-of-done-keys)."""
    rows, done = [], set()
    p = Path(path)
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add((int(row['L']), row['k'], int(row['epoch'])))
    return rows, done


def save_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_results(csv_path, output_prefix):
    """Generate two sets of plots from the results CSV."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'L': int(r['L']),
                'k': float(r['k']),
                'epoch': int(r['epoch']),
                'mean': float(r['accept_mean']),
                'std': float(r['accept_std']),
            })
    if not rows:
        print("No data to plot.")
        return

    Ls = sorted(set(r['L'] for r in rows))
    ks = sorted(set(r['k'] for r in rows))

    cmap_L = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ls)))
    cmap_k = plt.cm.plasma(np.linspace(0.15, 0.85, len(ks)))

    # --- Figure 1: one subplot per κ, curves = different L ---
    fig1, axes1 = plt.subplots(1, len(ks), figsize=(6 * len(ks), 5),
                                squeeze=False, sharey=True)
    for j, kv in enumerate(ks):
        ax = axes1[0, j]
        for i, Lv in enumerate(Ls):
            pts = sorted(
                [r for r in rows if r['L'] == Lv and r['k'] == kv],
                key=lambda r: r['epoch'],
            )
            if not pts:
                continue
            ep = [p['epoch'] for p in pts]
            mn = [p['mean'] for p in pts]
            sd = [p['std'] for p in pts]
            ax.errorbar(ep, mn, yerr=sd, label=f'L={Lv}',
                        color=cmap_L[i], marker='o', ms=3, lw=1.2,
                        capsize=2, elinewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_title(f'κ = {kv}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes1[0, 0].set_ylabel('MALA acceptance rate')
    fig1.suptitle('Acceptance rate vs epoch  (grouped by κ)', fontsize=13)
    fig1.tight_layout()
    fig1.savefig(f'{output_prefix}_by_kappa.png', dpi=200)
    print(f"Saved {output_prefix}_by_kappa.png")

    # --- Figure 2: one subplot per L, curves = different κ ---
    fig2, axes2 = plt.subplots(1, len(Ls), figsize=(5 * len(Ls), 5),
                                squeeze=False, sharey=True)
    for i, Lv in enumerate(Ls):
        ax = axes2[0, i]
        for j, kv in enumerate(ks):
            pts = sorted(
                [r for r in rows if r['L'] == Lv and r['k'] == kv],
                key=lambda r: r['epoch'],
            )
            if not pts:
                continue
            ep = [p['epoch'] for p in pts]
            mn = [p['mean'] for p in pts]
            sd = [p['std'] for p in pts]
            ax.errorbar(ep, mn, yerr=sd, label=f'κ={kv}',
                        color=cmap_k[j], marker='s', ms=3, lw=1.2,
                        capsize=2, elinewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_title(f'L = {Lv}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes2[0, 0].set_ylabel('MALA acceptance rate')
    fig2.suptitle('Acceptance rate vs epoch  (grouped by L)', fontsize=13)
    fig2.tight_layout()
    fig2.savefig(f'{output_prefix}_by_L.png', dpi=200)
    print(f"Saved {output_prefix}_by_L.png")

    plt.close('all')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MALA acceptance-rate sweep over (L, κ, epoch)."
    )
    # config selection
    parser.add_argument("--base_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    parser.add_argument("--L", type=int, nargs="+", default=None,
                        help="Lattice sizes to include (default: all)")
    parser.add_argument("--k", type=float, nargs="+", default=None,
                        help="Kappa values to include (default: all)")
    parser.add_argument("--network", type=str, default="ncsnpp")
    # MALA hyper-params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=50,
                        help="MALA iterations on reference data")
    parser.add_argument("--t_mh", type=float, default=0.01)
    parser.add_argument("--mala_step_size", type=float, default=None,
                        help="Langevin step h (default: 1.0/L²)")
    # epoch filtering
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--epoch_stride", type=int, default=1,
                        help="Only evaluate every N-th checkpoint")
    # output
    parser.add_argument("--output", type=str,
                        default=os.path.join(SCRIPT_DIR, "accept_rate_analysis"))
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip computation, re-plot from CSV")
    args = parser.parse_args()

    csv_path = f"{args.output}.csv"

    if args.plot_only:
        plot_results(csv_path, args.output)
        return

    # --- discover & filter ---
    configs = discover_configs(args.base_dir, args.network)
    if args.L:
        configs = [(L, k, l, d) for L, k, l, d in configs if L in args.L]
    if args.k:
        configs = [(L, k, l, d) for L, k, l, d in configs
                   if any(abs(k - kv) < 1e-6 for kv in args.k)]

    print(f"Configurations ({len(configs)}):")
    for L, k, l, d in configs:
        print(f"  L={L:>3d}  κ={k}  λ={l}  dir={d.name}")

    # --- load existing results ---
    results, done = load_csv(csv_path)
    if done:
        print(f"Resuming — {len(done)} evaluations already cached in {csv_path}")

    # --- sweep ---
    ref_cache = {}
    for L, k, l, model_dir in configs:
        ckpts = find_epoch_checkpoints(model_dir / "models")
        epochs = sorted(ckpts.keys())
        if args.max_epoch is not None:
            epochs = [e for e in epochs if e <= args.max_epoch]
        epochs = epochs[::args.epoch_stride]

        print(f"\n{'='*60}")
        print(f"L={L}, κ={k}, λ={l}  —  {len(epochs)} checkpoints")
        print(f"{'='*60}")

        # Load reference data once per (L, k) combination
        ref_key = (L, k, l)
        if ref_key not in ref_cache:
            any_ckpt = torch.load(ckpts[epochs[0]], map_location="cpu",
                                  weights_only=False)
            hp = any_ckpt.get("hyper_parameters", {})
            norm_min = hp.get("norm_min") or -6.22
            norm_max = hp.get("norm_max") or 6.19
            del any_ckpt

            data_path = os.path.join(
                args.base_dir,
                f"trainingdata/cfgs_wolff_fahmc_k={k}_l={l}_{L}^2.jld2"
            )
            print(f"  Loading reference data: {data_path}")
            ref_cache[ref_key] = load_reference_data(
                data_path, args.num_samples, norm_min, norm_max, args.device
            )
            print(f"  Loaded {ref_cache[ref_key].shape[0]} configs")

        x_ref = ref_cache[ref_key]

        for epoch in epochs:
            key = (L, str(k), epoch)
            if key in done:
                print(f"  epoch {epoch:5d}  [cached]")
                continue

            ckpt_path = ckpts[epoch]
            print(f"  epoch {epoch:5d}  {ckpt_path.name} ... ",
                  end="", flush=True)
            try:
                mean_acc, std_acc = run_mala_diagnostic(
                    ckpt_path, args.device, x_ref,
                    mh_steps=args.mh_steps,
                    t_mh=args.t_mh,
                    mala_step_size=args.mala_step_size,
                )
                print(f"accept = {mean_acc:.4f} ± {std_acc:.4f}")
                results.append({
                    'L': L, 'k': k, 'l': l,
                    'epoch': epoch,
                    'accept_mean': f"{mean_acc:.6f}",
                    'accept_std': f"{std_acc:.6f}",
                })
                done.add(key)
                save_csv(results, csv_path)
            except Exception as e:
                print(f"FAILED — {e}")

    # --- plot ---
    print(f"\nAll done.  Total evaluations: {len(results)}")
    plot_results(csv_path, args.output)


if __name__ == "__main__":
    main()
