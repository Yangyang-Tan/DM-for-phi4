"""
Generate CIFAR-10 samples at different reverse-SDE stopping times t0.

Runs the reverse SDE from t=1 once and saves snapshots at multiple t0 values.
This allows studying how the propagator G(k) converges as t0 -> 0:
  - IR modes (low k) should align first (large t0)
  - UV modes (high k) should align last (small t0)

Usage:
    python sample_at_t0_cifar10.py --ep "epoch=0499" --class_name cat --device cuda:0
    python sample_at_t0_cifar10.py --ep "epoch=0999" --t0_list 0.9,0.7,0.5,0.3,0.1,0.05,0.01
"""

import sys
sys.path.append("..")

import functools
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from networks import ScoreNetUNet, NCSNpp2D
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
    parser = argparse.ArgumentParser(
        description="Sample CIFAR-10 at different reverse-SDE stopping times")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--ep", type=str, default=None)
    parser.add_argument("--class_name", type=str, default="cat")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["unet", "ncsnpp"])
    parser.add_argument("--schedule", type=str, default="log")
    parser.add_argument("--n_repeats", type=int, default=4,
                        help="Number of independent runs to concatenate")
    parser.add_argument("--t0_list", type=str,
                        default="0.9,0.7,0.5,0.3,0.2,0.1,0.05,0.03,0.01",
                        help="Comma-separated list of stopping times")
    args = parser.parse_args()

    t0_list = [float(x) for x in args.t0_list.split(",")]
    print(f"Stopping times: {t0_list}")

    model_dir = f"cifar10_{args.class_name}_{args.network}"

    # Auto-discover checkpoint
    if args.checkpoint is None:
        pattern = f"*{args.ep}*.ckpt" if args.ep else "*.ckpt"
        ckpts = sorted(Path(f"{model_dir}/models").glob(pattern))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found matching '{pattern}'")
        args.checkpoint = str(ckpts[-1])
    print(f"Checkpoint: {args.checkpoint}")

    # Load hparams
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 25.0)
    print(f"sigma={sigma}")

    # Create score model
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn, periodic=False)
    else:
        score_model = ScoreNetUNet(marginal_prob_std_fn, image_size=32)

    model = DiffusionModel.load_from_checkpoint(args.checkpoint, score_model=score_model)
    model = model.to(args.device).eval()

    out_dir = Path(f"{model_dir}/data_t0")
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

    ep_label = args.ep if args.ep else "latest"
    for t0, arrays in all_snapshots.items():
        samples_np = np.concatenate(arrays, axis=0)
        samples_out = samples_np.transpose(1, 2, 0)  # (32, 32, N)
        fname = out_dir / f"samples_{ep_label}_t0={t0:.4f}.npy"
        np.save(fname, samples_out)
        print(f"Saved {fname}  shape={samples_out.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
