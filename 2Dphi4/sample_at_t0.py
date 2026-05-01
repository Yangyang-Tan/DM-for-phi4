"""
Generate field configurations at different reverse-SDE stopping times t0.

Runs the reverse SDE from t=1 once and saves snapshots at multiple t0 values.
This allows studying how the propagator G(k) converges as t0 → 0:
  - IR modes (low k) should align first (large t0)
  - UV modes (high k) should align last (small t0)

Usage:
    python sample_at_t0.py --ep "epoch=0999" --k 0.2705 --device cuda:3
    python sample_at_t0.py --ep "epoch=0999" --k 0.2705 --t0_list 0.9,0.7,0.5,0.3,0.1,0.05,0.01
"""

import sys
sys.path.append("..")

import re
import functools
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from networks import NCSNpp2D, ScoreNet, ScoreNetUNetPeriodic
from diffusion_lightning import DiffusionModel, marginal_prob_std


def sample_snapshots(model, num_samples, num_steps, t0_list, schedule='log', eps=1e-5):
    """Run reverse SDE once and save snapshots at each t0 in t0_list.

    Returns:
        dict: {t0_value: tensor of shape (num_samples, 1, L, L)}
    """
    model.eval()
    device = model.device
    t0_sorted = sorted(t0_list, reverse=True)

    with model.ema.average_parameters():
        time_steps = model._build_time_steps(num_steps, eps, schedule, device)

        init_std = model.marginal_prob_std_fn(torch.tensor(1.0, device=device))
        x = torch.randn(num_samples, 1, model.L, model.L, device=device) * init_std

        snapshots = {}
        next_t0_idx = 0

        for i in tqdm(range(num_steps), desc="Sampling"):
            time_step = time_steps[i]
            next_time = time_steps[i + 1]

            while next_t0_idx < len(t0_sorted) and time_step >= t0_sorted[next_t0_idx] > next_time:
                t0_val = t0_sorted[next_t0_idx]
                snapshots[t0_val] = x.clone()
                print(f"  snapshot at t0={t0_val:.4f} (SDE step {i}, t={time_step.item():.4f})")
                next_t0_idx += 1

            dt = time_step - next_time
            batch_t = torch.ones(num_samples, device=device) * time_step
            g = model.diffusion_coeff_fn(time_step)
            mean_x = x + g**2 * model(x, batch_t) * dt
            x = mean_x + g * torch.sqrt(dt) * torch.randn_like(x)

        snapshots[0.0] = mean_x.clone()
        print(f"  snapshot at t0=0.0000 (final)")

    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--ep", type=str, default=None)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--k", type=float, default=0.2705)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["scorenet", "unet", "ncsnpp"])
    parser.add_argument("--schedule", type=str, default="log")
    parser.add_argument("--n_repeats", type=int, default=4,
                        help="Number of independent runs to concatenate")
    parser.add_argument("--t0_list", type=str,
                        default="0.9,0.7,0.5,0.3,0.2,0.1,0.05,0.03,0.01",
                        help="Comma-separated list of stopping times")
    args = parser.parse_args()

    t0_list = [float(x) for x in args.t0_list.split(",")]
    print(f"Stopping times: {t0_list}")

    if args.checkpoint is None:
        ckpts = sorted(Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}/models").glob(f"*{args.ep}*.ckpt"))
        args.checkpoint = str(ckpts[-1]) if ckpts else None
    print(f"Checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"sigma={sigma}, norm=[{norm_min:.4f}, {norm_max:.4f}]")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn)
    elif args.network == "scorenet":
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)

    model = DiffusionModel.load_from_checkpoint(args.checkpoint, score_model=score_model)
    model = model.to(args.device).eval()

    out_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}/data_t0")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_snapshots = {t0: [] for t0 in t0_list}
    all_snapshots[0.0] = []

    for run_idx in range(args.n_repeats):
        print(f"\n=== Run {run_idx+1}/{args.n_repeats} ===")
        with torch.no_grad():
            snapshots = sample_snapshots(
                model, args.num_samples, args.num_steps,
                t0_list, schedule=args.schedule
            )
        for t0, tensor in snapshots.items():
            all_snapshots[t0].append(tensor[:, 0].cpu().numpy())

    for t0, arrays in all_snapshots.items():
        samples_norm = np.concatenate(arrays, axis=0)
        samples_renorm = (samples_norm + 1) / 2 * (norm_max - norm_min) + norm_min
        samples_out = samples_renorm.transpose(1, 2, 0)  # (L, L, N)
        fname = out_dir / f"samples_{args.ep}_t0={t0:.4f}.npy"
        np.save(fname, samples_out)
        print(f"Saved {fname}  shape={samples_out.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
