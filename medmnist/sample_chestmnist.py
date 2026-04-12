"""
Generate samples from a trained ChestMNIST diffusion model.

Usage:
    python sample_chestmnist.py
    python sample_chestmnist.py --num_samples 64 --method pc
"""

import sys
sys.path.append("..")

import functools
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from networks import ScoreNetUNet
from diffusion_lightning import DiffusionModel, marginal_prob_std


def get_latest_checkpoint(model_dir="models/chestmnist"):
    """Get the latest checkpoint file from the model directory."""
    ckpt_files = sorted(Path(model_dir).glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    return str(ckpt_files[-1])


def main():
    parser = argparse.ArgumentParser(description="Sample from trained ChestMNIST model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--num_steps", type=int, default=2000, help="Sampling steps")
    parser.add_argument("--method", type=str, default="em", choices=["em", "pc"], help="Sampling method")
    parser.add_argument("--output", type=str, default="chestmnist_samples.png", help="Output image path")
    parser.add_argument("--sigma", type=float, default=25.0, help="Noise scale (must match training)")
    parser.add_argument("--size", type=int, default=64, help="Image size")
    parser.add_argument("--device", type=str, default="cuda:2", help="GPU device")
    args = parser.parse_args()

    # Get checkpoint path
    checkpoint = args.checkpoint if args.checkpoint else get_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint}")

    device = args.device

    # Recreate model architecture
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    score_model = ScoreNetUNet(marginal_prob_std_fn, image_size=args.size)
    
    # Load model from checkpoint
    model = DiffusionModel.load_from_checkpoint(
        checkpoint,
        score_model=score_model,
        sigma=args.sigma,
        L=args.size,
    )
    model = model.to(device)
    model.eval()

    # Generate samples
    print(f"Generating {args.num_samples} samples using {args.method.upper()} sampler...")
    if args.method == "em":
        samples = model.sample(num_samples=args.num_samples, num_steps=args.num_steps)
    else:
        samples = model.sample_pc(num_samples=args.num_samples, num_steps=args.num_steps)

    # Plot samples
    samples = samples.clamp(0.0, 1.0)
    n_rows = int(args.num_samples ** 0.5)
    n_cols = (args.num_samples + n_rows - 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten() if args.num_samples > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < args.num_samples:
            ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Samples saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
