"""
Analyze score quality vs training epoch for different (L, κ) configs.

Compares model score s(x, t) against the true −∇S on training data.
Metrics: cosine similarity, relative MSE, magnitude ratio.

Usage:
    python analyze_score_quality.py --device cuda:0
    python analyze_score_quality.py --device cuda:0 --L 64 128 --k 0.28
    python analyze_score_quality.py --plot_only
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import io
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
from sample_phi4 import phi4_grad_S


# ---------------------------------------------------------------------------
#  Discovery helpers (shared with analyze_accept_rate.py)
# ---------------------------------------------------------------------------

def discover_configs(base_dir, network="ncsnpp"):
    configs = []
    for d in sorted(Path(base_dir).iterdir()):
        m = re.match(
            rf'phi4_L(\d+)_k([\d.]+)_l([\d.]+)_{re.escape(network)}$', d.name
        )
        if m and d.is_dir():
            configs.append((int(m.group(1)), float(m.group(2)),
                            float(m.group(3)), d))
    configs.sort(key=lambda x: (x[0], x[1]))
    return configs


def find_epoch_checkpoints(model_dir):
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=epoch=*.ckpt")):
        if '-v' in f.stem:
            continue
        m = re.search(r'epoch=(\d+)', f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.default_rng(42).choice(cfgs.shape[0], size=num_samples,
                                            replace=False)
    cfgs_norm = ((cfgs[idx] - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


# ---------------------------------------------------------------------------
#  Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def run_score_quality(ckpt_path, device, x_ref, t_eval=0.01,
                      denoise_t=None):
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

    # Handle torch.compile _orig_mod. prefix in state_dict
    ckpt2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt2.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt2["state_dict"] = {
            key.replace("._orig_mod.", "."): v for key, v in sd.items()
        }
        buf = io.BytesIO()
        torch.save(ckpt2, buf)
        del ckpt2
        buf.seek(0)
        model = DiffusionModel.load_from_checkpoint(buf, score_model=score_model)
    else:
        del ckpt2
        model = DiffusionModel.load_from_checkpoint(
            str(ckpt_path), score_model=score_model
        )
    model = model.to(device).eval()

    grad_S_fn = functools.partial(phi4_grad_S, k=k, l=l,
                                  phi_min=norm_min, phi_max=norm_max)

    metrics = model.score_quality(x_ref, grad_S_fn, t_eval=t_eval)
    result = {k_: v.mean().item() for k_, v in metrics.items()}

    # Denoising score matching loss at multiple t
    if denoise_t is not None:
        dsm = model.denoising_score_eval(x_ref, t_values=denoise_t)
        for t_val, loss in dsm.items():
            result[f'dsm_{t_val}'] = loss

    del model, score_model
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
#  CSV I/O
# ---------------------------------------------------------------------------

BASE_FIELDS = ['L', 'k', 'l', 'epoch', 'cos_sim', 'rel_mse', 'mag_ratio']


def load_csv(path):
    rows, done = [], set()
    p = Path(path)
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add((int(row['L']), row['k'], int(row['epoch'])))
    return rows, done


def get_fields(denoise_t):
    fields = list(BASE_FIELDS)
    if denoise_t:
        fields += [f'dsm_{t}' for t in denoise_t]
    return fields


def save_csv(rows, path, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_results(csv_path, output_prefix):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'L': int(r['L']), 'k': float(r['k']),
                'epoch': int(r['epoch']),
                'cos_sim': float(r['cos_sim']),
                'rel_mse': float(r['rel_mse']),
                'mag_ratio': float(r['mag_ratio']),
            })
    if not rows:
        print("No data to plot.")
        return

    Ls = sorted(set(r['L'] for r in rows))
    ks = sorted(set(r['k'] for r in rows))
    cmap_L = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(Ls), 2)))
    cmap_k = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(ks), 2)))

    metrics = [('cos_sim', 'Cosine similarity  ⟨s_model, −∇S⟩'),
               ('rel_mse', 'Relative MSE  ||s−(−∇S)||²/||∇S||²'),
               ('mag_ratio', 'Magnitude ratio  |s_model|/|∇S|')]

    # --- Figure 1: grouped by κ ---
    fig1, axes = plt.subplots(len(metrics), len(ks),
                               figsize=(5 * len(ks), 4 * len(metrics)),
                               squeeze=False, sharex='col')
    for col, kv in enumerate(ks):
        for row, (mkey, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            for i, Lv in enumerate(Ls):
                pts = sorted([r for r in rows
                              if r['L'] == Lv and r['k'] == kv],
                             key=lambda r: r['epoch'])
                if not pts:
                    continue
                ax.plot([p['epoch'] for p in pts],
                        [p[mkey] for p in pts],
                        label=f'L={Lv}', color=cmap_L[i],
                        marker='o', ms=2.5, lw=1)
            if row == 0:
                ax.set_title(f'κ = {kv}')
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            if row == len(metrics) - 1:
                ax.set_xlabel('Epoch')
    fig1.suptitle('Score quality vs epoch  (grouped by κ)', fontsize=13, y=1.01)
    fig1.tight_layout()
    fig1.savefig(f'{output_prefix}_by_kappa.png', dpi=200, bbox_inches='tight')
    print(f"Saved {output_prefix}_by_kappa.png")

    # --- Figure 2: grouped by L ---
    fig2, axes = plt.subplots(len(metrics), len(Ls),
                               figsize=(5 * len(Ls), 4 * len(metrics)),
                               squeeze=False, sharex='col')
    for col, Lv in enumerate(Ls):
        for row, (mkey, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            for j, kv in enumerate(ks):
                pts = sorted([r for r in rows
                              if r['L'] == Lv and r['k'] == kv],
                             key=lambda r: r['epoch'])
                if not pts:
                    continue
                ax.plot([p['epoch'] for p in pts],
                        [p[mkey] for p in pts],
                        label=f'κ={kv}', color=cmap_k[j],
                        marker='s', ms=2.5, lw=1)
            if row == 0:
                ax.set_title(f'L = {Lv}')
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            if row == len(metrics) - 1:
                ax.set_xlabel('Epoch')
    fig2.suptitle('Score quality vs epoch  (grouped by L)', fontsize=13, y=1.01)
    fig2.tight_layout()
    fig2.savefig(f'{output_prefix}_by_L.png', dpi=200, bbox_inches='tight')
    print(f"Saved {output_prefix}_by_L.png")

    plt.close('all')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score-quality sweep over (L, κ, epoch)."
    )
    parser.add_argument("--base_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    parser.add_argument("--L", type=int, nargs="+", default=None)
    parser.add_argument("--k", type=float, nargs="+", default=None)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--t_eval", type=float, default=0.01,
                        help="Time at which model score is evaluated")
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--epoch_stride", type=int, default=1)
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "score_quality"))
    parser.add_argument("--denoise_t", type=float, nargs="+", default=None,
                        help="Time points for denoising loss eval "
                             "(e.g. --denoise_t 0.01 0.1 0.5)")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    csv_path = f"{args.output}.csv"

    if args.plot_only:
        plot_results(csv_path, args.output)
        return

    configs = discover_configs(args.base_dir, args.network)
    if args.L:
        configs = [(L, k, l, d) for L, k, l, d in configs if L in args.L]
    if args.k:
        configs = [(L, k, l, d) for L, k, l, d in configs
                   if any(abs(k - kv) < 1e-6 for kv in args.k)]

    print(f"Configurations ({len(configs)}):")
    for L, k, l, d in configs:
        print(f"  L={L:>3d}  κ={k}  λ={l}  dir={d.name}")

    fields = get_fields(args.denoise_t)
    results, done = load_csv(csv_path)
    if done:
        print(f"Resuming — {len(done)} cached in {csv_path}")

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
            print(f"  Loading: {data_path}")
            ref_cache[ref_key] = load_reference_data(
                data_path, args.num_samples, norm_min, norm_max, args.device
            )

        x_ref = ref_cache[ref_key]

        for epoch in epochs:
            key = (L, str(k), epoch)
            if key in done:
                print(f"  epoch {epoch:5d}  [cached]")
                continue

            ckpt_path = ckpts[epoch]
            print(f"  epoch {epoch:5d}  ... ", end="", flush=True)
            try:
                m = run_score_quality(ckpt_path, args.device, x_ref,
                                      t_eval=args.t_eval,
                                      denoise_t=args.denoise_t)
                line = (f"cos={m['cos_sim']:.4f}  "
                        f"relMSE={m['rel_mse']:.4f}  "
                        f"mag={m['mag_ratio']:.4f}")
                if args.denoise_t:
                    dsm_parts = [f"dsm@{t}={m[f'dsm_{t}']:.2f}"
                                 for t in args.denoise_t]
                    line += "  " + "  ".join(dsm_parts)
                print(line)
                row = {
                    'L': L, 'k': k, 'l': l, 'epoch': epoch,
                    'cos_sim': f"{m['cos_sim']:.6f}",
                    'rel_mse': f"{m['rel_mse']:.6f}",
                    'mag_ratio': f"{m['mag_ratio']:.6f}",
                }
                if args.denoise_t:
                    for t in args.denoise_t:
                        row[f'dsm_{t}'] = f"{m[f'dsm_{t}']:.6f}"
                results.append(row)
                done.add(key)
                save_csv(results, csv_path, fields)
            except Exception as e:
                print(f"FAILED — {e}")

    print(f"\nDone. Total: {len(results)}")
    plot_results(csv_path, args.output)


if __name__ == "__main__":
    main()
