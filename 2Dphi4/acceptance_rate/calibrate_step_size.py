"""
Calibrate MALA step size by sweeping h on an early vs late checkpoint.

Uses *fixed* training-data configurations so that all models start from
the same reference point.  The acceptance rate then purely reflects
score quality (no confounding from sample location).

Usage:
    python calibrate_step_size.py --device cuda:0 --L 128 --k 0.28
    python calibrate_step_size.py --device cuda:0 --L 128 --k 0.28 \
        --early_epoch 9 --late_epoch 1999
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import re
import functools
import argparse
from pathlib import Path

import io

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from sample_phi4 import phi4_action


def find_epoch_checkpoints(model_dir):
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=epoch=*.ckpt")):
        if '-v' in f.stem:
            continue
        m = re.search(r'epoch=(\d+)', f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19

    path_str = str(ckpt_path)
    k = float(re.search(r'_k([\d.]+)', path_str).group(1))
    l = float(re.search(r'_l([\d.]+)', path_str).group(1))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel.load_from_checkpoint(buf, score_model=score_model)
    model = model.to(device).eval()
    action_fn = functools.partial(
        phi4_action, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )
    return model, action_fn


def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    """Load training data, normalise to [−1, 1], return (num_samples, 1, L, L)."""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def sweep_one(model, action_fn, h_values, x_ref, mh_steps, t_mh):
    """Run MALA at each step size on fixed reference data."""
    results = []
    device = model.device
    num_samples = x_ref.shape[0]

    for h_val in h_values:
        x = x_ref.clone()
        h = torch.tensor(h_val, device=device, dtype=x.dtype)
        batch_t = torch.ones(num_samples, device=device) * t_mh
        total_accept = torch.zeros(num_samples, device=device)

        with torch.no_grad():
            for _ in range(mh_steps):
                score_x = model(x, batch_t)
                noise = torch.randn_like(x)
                y = x + h * score_x + torch.sqrt(2 * h) * noise

                score_y = model(y, batch_t)
                drift = h * (score_x + score_y)
                log_q = -(0.25 / h) * torch.sum(
                    (torch.sqrt(2 * h) * noise + drift) ** 2, dim=(1, 2, 3)
                ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3))
                log_pi = action_fn(x) - action_fn(y)

                accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
                accept = torch.rand(num_samples, device=device) < accept_prob
                x = torch.where(accept[:, None, None, None], y, x)
                total_accept += accept.float()

        acc = total_accept / mh_steps
        results.append((acc.mean().item(), (acc.std() / acc.shape[0] ** 0.5).item()))
        print(f"    h = {h_val:.2e}  →  accept = {acc.mean().item():.4f} "
              f"± {acc.std().item():.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--early_epoch", type=int, default=None,
                        help="Epoch for 'bad' model (default: first available)")
    parser.add_argument("--late_epoch", type=int, default=None,
                        help="Epoch for 'good' model (default: last available)")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=20)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Training data .jld2 (default: auto-detect)")
    args = parser.parse_args()

    model_dir = Path(f"phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}")
    ckpts = find_epoch_checkpoints(model_dir / "models")
    epochs = sorted(ckpts.keys())

    early_ep = args.early_epoch or epochs[0]
    late_ep = args.late_epoch or epochs[-1]
    print(f"Early checkpoint: epoch {early_ep}")
    print(f"Late  checkpoint: epoch {late_ep}")

    # Load norm parameters from any checkpoint
    ref_ckpt = torch.load(ckpts[early_ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    del ref_ckpt
    print(f"norm range: [{norm_min}, {norm_max}]")

    # Load reference training data
    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    print(f"Reference data: {args.data_path}")
    x_ref = load_reference_data(
        args.data_path, args.num_samples, norm_min, norm_max, args.device
    )
    print(f"Loaded {x_ref.shape[0]} reference configs, shape {x_ref.shape}")

    L = args.L
    c_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    h_values = c_values / (L ** 2)

    print(f"\nSweeping c = h·L²  over {c_values.tolist()}")
    print(f"(h = c / {L}² = c / {L**2})\n")

    # --- early (bad) model ---
    print(f"{'='*50}")
    print(f"Early model  (epoch {early_ep})")
    print(f"{'='*50}")
    model_early, action_fn = load_model(ckpts[early_ep], args.device)
    res_early = sweep_one(model_early, action_fn, h_values, x_ref,
                          args.mh_steps, args.t_mh)
    del model_early
    torch.cuda.empty_cache()

    # --- late (good) model ---
    print(f"\n{'='*50}")
    print(f"Late model   (epoch {late_ep})")
    print(f"{'='*50}")
    model_late, action_fn = load_model(ckpts[late_ep], args.device)
    res_late = sweep_one(model_late, action_fn, h_values, x_ref,
                         args.mh_steps, args.t_mh)
    del model_late
    torch.cuda.empty_cache()

    # --- analysis ---
    acc_early = np.array([r[0] for r in res_early])
    acc_late = np.array([r[0] for r in res_late])
    gap = acc_late - acc_early

    print(f"\n{'='*50}")
    print(f"{'c':>8s}  {'h':>10s}  {'early':>7s}  {'late':>7s}  {'gap':>7s}")
    print(f"{'='*50}")
    for i, c in enumerate(c_values):
        print(f"{c:8.2f}  {h_values[i]:10.2e}  "
              f"{acc_early[i]:7.4f}  {acc_late[i]:7.4f}  {gap[i]:+7.4f}")

    useful = (acc_late > 0.3) & (acc_late < 0.85)
    if useful.any():
        best_idx = np.where(useful, gap, -np.inf).argmax()
    else:
        best_idx = gap.argmax()
    print(f"\nRecommended:  c = {c_values[best_idx]:.2f}  "
          f"(h = {h_values[best_idx]:.2e})")
    print(f"  early accept = {acc_early[best_idx]:.4f}")
    print(f"  late  accept = {acc_late[best_idx]:.4f}")
    print(f"  gap          = {gap[best_idx]:+.4f}")

    # --- plot (publication quality) ---
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(c_values, acc_early,
                 yerr=[r[1] for r in res_early],
                 label=f'Epoch {early_ep} (early)', marker='o',
                 capsize=3, color='C3')
    ax1.errorbar(c_values, acc_late,
                 yerr=[r[1] for r in res_late],
                 label=f'Epoch {late_ep} (late)', marker='s',
                 capsize=3, color='C0')
    ax1.axvline(c_values[best_idx], ls='--', color='gray', alpha=0.6,
                label=f'Recommended $c={c_values[best_idx]:.1f}$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$c\;(h = c / L^2)$')
    ax1.set_ylabel('Acceptance rate')
    t_exp = int(np.floor(np.log10(args.t_mh)))
    t_man = args.t_mh / 10**t_exp
    t_str = rf'10^{{{t_exp}}}' if t_man == 1 else rf'{t_man:.0f}\times10^{{{t_exp}}}'
    ax1.set_title(rf'$L={L},\;\kappa={args.k},\;t_{{\mathrm{{mh}}}}={t_str}$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    ax2.bar(range(len(c_values)), gap, tick_label=[f'{c:.2g}' for c in c_values],
            color=['C2' if useful[i] else 'C7' for i in range(len(c_values))])
    ax2.set_xlabel(r'$c$')
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, f"calibrate_h_L{L}_k{args.k}.pdf")
    fig.savefig(out_path)
    print(f"\nSaved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
