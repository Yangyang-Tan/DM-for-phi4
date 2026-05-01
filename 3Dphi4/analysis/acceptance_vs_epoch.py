"""
Track MALA acceptance rate vs training epoch for 3D phi4.

Usage:
    python acceptance_vs_epoch.py --device cuda:0 --L 64 --k 0.2
    python acceptance_vs_epoch.py --device cuda:0 --L 64 --k 0.1923 --every 100
"""

import sys
import re
import functools
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))   # 3Dphi4/
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # repo root

import io

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from sample_phi4 import phi4_action_3d


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
    score_model = NCSNpp3D(marginal_prob_std_fn)

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel3D.load_from_checkpoint(buf, score_model=score_model)
    model = model.to(device).eval()
    action_fn = functools.partial(
        phi4_action_3d, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )
    return model, action_fn


def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    """Load training data, normalise to [-1, 1], return (num_samples, 1, L, L, L)."""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 4:
        x = x.unsqueeze(1)  # (N, L, L, L) -> (N, 1, L, L, L)
    return x


def measure_acceptance(model, action_fn, x_ref, h, t_mh, mh_steps, batch_size=128):
    """Run MALA on fixed reference data and return (mean_acc, std_acc)."""
    device = model.device
    num_samples = x_ref.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x_ref.dtype)
    x = x_ref.clone()
    total_accept = torch.zeros(num_samples, device=device)

    with torch.no_grad():
        for _ in range(mh_steps):
            score_x = torch.cat([model(x[i:i+batch_size], torch.ones(min(batch_size, num_samples-i), device=device)*t_mh) for i in range(0, num_samples, batch_size)])
            noise = torch.randn_like(x)
            y = x + h_t * score_x + torch.sqrt(2 * h_t) * noise

            score_y = torch.cat([model(y[i:i+batch_size], torch.ones(min(batch_size, num_samples-i), device=device)*t_mh) for i in range(0, num_samples, batch_size)])
            drift = h_t * (score_x + score_y)
            log_q = -(0.25 / h_t) * torch.sum(
                (torch.sqrt(2 * h_t) * noise + drift) ** 2, dim=(1, 2, 3, 4)
            ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3, 4))
            log_pi = action_fn(x) - action_fn(y)

            accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
            accept = torch.rand(num_samples, device=device) < accept_prob
            x = torch.where(accept[:, None, None, None, None], y, x)
            total_accept += accept.float()

    acc = total_accept / mh_steps
    return acc.mean().item(), (acc.std() / acc.shape[0] ** 0.5).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=1)
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, nargs='*', default=None)
    parser.add_argument("--every", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path(f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_{args.network}")
    ckpts = find_epoch_checkpoints(model_dir / "models")
    all_epochs = sorted(ckpts.keys())

    if args.epochs:
        epochs = [e for e in args.epochs if e in ckpts]
    elif args.every:
        step = args.every
        target = all_epochs[0]
        epochs = []
        while target <= all_epochs[-1]:
            closest = min(all_epochs, key=lambda e: abs(e - target))
            if closest not in epochs:
                epochs.append(closest)
            target += step
        if all_epochs[-1] not in epochs:
            epochs.append(all_epochs[-1])
    else:
        epochs = all_epochs

    print(f"Found {len(ckpts)} checkpoints, evaluating {len(epochs)} epochs")
    print(f"Parameters: c = {args.c}, t_mh = {args.t_mh:.1e}, "
          f"h = {args.c / args.L**3:.2e}")

    # Load norm parameters
    first_ckpt = torch.load(ckpts[epochs[0]], map_location="cpu", weights_only=False)
    hparams = first_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"norm range: [{norm_min}, {norm_max}]")
    del first_ckpt

    # Load reference data
    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"
    np.random.seed(42)
    x_ref = load_reference_data(
        args.data_path, args.num_samples, norm_min, norm_max, args.device
    )
    print(f"Reference data: {args.data_path}, shape {x_ref.shape}")

    L = args.L
    h = args.c / (L ** 3)

    # Sweep epochs
    acc_means = []
    acc_stds = []

    for ep in epochs:
        print(f"\nEpoch {ep:>6d}: ", end="", flush=True)
        model, action_fn = load_model(ckpts[ep], args.device)
        mean, std = measure_acceptance(model, action_fn, x_ref, h, args.t_mh,
                                       args.mh_steps)
        acc_means.append(mean)
        acc_stds.append(std)
        print(f"accept = {mean:.4f} ± {std:.4f}")
        del model
        torch.cuda.empty_cache()

    # Save results as CSV
    epochs_arr = np.array(epochs)
    acc_means_arr = np.array(acc_means)
    acc_stds_arr = np.array(acc_stds)

    out_csv = f"acceptance_vs_epoch_3d_L{L}_k{args.k}.csv"
    with open(out_csv, 'w') as f:
        f.write(f"# c={args.c}, t_mh={args.t_mh}, h={h}, L={L}, k={args.k}\n")
        f.write("epoch,acc_mean,acc_std\n")
        for i, ep in enumerate(epochs):
            f.write(f"{ep},{acc_means_arr[i]:.6f},{acc_stds_arr[i]:.6f}\n")
    print(f"\nSaved data to {out_csv}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"{'Epoch':>8s}  {'Accept':>10s}")
    print(f"{'='*50}")
    for i, ep in enumerate(epochs):
        print(f"{ep:8d}  {acc_means_arr[i]:7.4f} ± {acc_stds_arr[i]:.4f}")

    # Plot (publication quality)
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 4,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, ax = plt.subplots(figsize=(7, 4.9))
    ax.errorbar(epochs_arr, acc_means_arr, yerr=acc_stds_arr,
                marker='o', capsize=2, markersize=4, linewidth=1.2, color='C0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Acceptance rate')
    ax.set_title(rf'3D $\phi^4$: $L={L},\;\kappa={args.k}$')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_plot = f"acceptance_vs_epoch_3d_L{L}_k{args.k}.pdf"
    fig.savefig(out_plot)
    print(f"Saved plot to {out_plot}")
    plt.close()


if __name__ == "__main__":
    main()
