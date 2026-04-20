"""
Generate samples from a trained CIFAR-10 diffusion model.

Saves samples as (Lx, Ly, N) .npy files for propagator analysis.

Usage:
    python sample_cifar10.py --ep "epoch=0499" --class_name cat --network ncsnpp
    python sample_cifar10.py --ep "epoch=0999" --method pc --num_samples 2048
"""

import sys
sys.path.append("..")

import os
import functools
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import ScoreNetUNet, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


def main():
    parser = argparse.ArgumentParser(description="Sample from trained CIFAR-10 diffusion model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (auto-discovers if None)")
    parser.add_argument("--num_samples", type=int, default=1024,
                        help="Samples per repeat")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of reverse SDE steps")
    parser.add_argument("--method", type=str, default="em",
                        choices=["em", "pc", "dpm1", "dpm2", "dpm3", "rk45"],
                        help="em (SDE), pc, dpm1/dpm2/dpm3 (DPM-Solver ODE), rk45")
    parser.add_argument("--ep", type=str, default=None,
                        help="Epoch pattern for checkpoint auto-discovery")
    parser.add_argument("--class_name", type=str, default="cat")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["unet", "ncsnpp"])
    parser.add_argument("--schedule", type=str, default="log",
                        help="Time step schedule: linear, quadratic, cosine, log, power_N")
    parser.add_argument("--n_repeats", type=int, default=4,
                        help="Number of independent sampling runs to concatenate")
    parser.add_argument("--plot_grid", type=int, default=8,
                        help="Grid size for visualization")
    args = parser.parse_args()

    model_dir = f"cifar10_{args.class_name}_{args.network}"

    # Auto-discover checkpoint
    if args.checkpoint is None:
        pattern = f"*{args.ep}*.ckpt" if args.ep else "*.ckpt"
        ckpts = sorted(Path(f"{model_dir}/models").glob(pattern))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}/models/ matching '{pattern}'")
        args.checkpoint = str(ckpts[-1])
    print(f"Checkpoint: {args.checkpoint}")

    # Load hparams from checkpoint
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

    # Load model
    model = DiffusionModel.load_from_checkpoint(args.checkpoint, score_model=score_model)
    model = model.to(args.device).eval()

    # Sample with multiple repeats
    all_samples = []
    for run_idx in range(args.n_repeats):
        print(f"\n=== Run {run_idx+1}/{args.n_repeats} ===")
        with torch.no_grad():
            if args.method == "em":
                samples = model.sample(
                    args.num_samples, args.num_steps, schedule=args.schedule)
            elif args.method == "pc":
                samples = model.sample_pc(
                    args.num_samples, args.num_steps, schedule=args.schedule)
            elif args.method in ("dpm1", "dpm2", "dpm3", "rk45"):
                samples = model.sample_ode(
                    args.num_samples, args.num_steps,
                    schedule=args.schedule, method=args.method)
        all_samples.append(samples[:, 0].cpu().numpy())

    samples_np = np.concatenate(all_samples, axis=0)  # (N_total, 32, 32)

    # Save as (Lx, Ly, N) for correlation analysis
    output_dir = f"{model_dir}/data"
    os.makedirs(output_dir, exist_ok=True)

    ep_label = args.ep if args.ep else "latest"
    samples_out = samples_np.transpose(1, 2, 0)  # (32, 32, N)
    output_path = f"{output_dir}/samples_epoch={ep_label}.npy"
    np.save(output_path, samples_out)
    print(f"\nSaved samples: {output_path}, shape={samples_out.shape}")
    print(f"  Range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")

    # Visualization
    n = min(args.plot_grid, int(np.sqrt(len(samples_np))))
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5))
    for i, ax in enumerate(axes.flatten()):
        # Map from [-1, 1] to [0, 1] for display
        img = (samples_np[i] + 1) / 2
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.suptitle(f'CIFAR-10 {args.class_name} — {args.method.upper()} epoch={ep_label}')
    plt.tight_layout()

    fig_dir = f"{model_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = f"{fig_dir}/samples_epoch={ep_label}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {fig_path}")


if __name__ == "__main__":
    main()
