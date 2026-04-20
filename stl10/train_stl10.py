"""
Train a score-based diffusion model on STL-10 grayscale images (64x64).

Designed for propagator analysis: saves periodic checkpoints and training
data in (Lx, Ly, N) format for correlation_stl10.jl.

Usage:
    # Unlabeled split (100k images, recommended)
    python train_stl10.py --split unlabeled --network ncsnpp --device cuda:0

    # Labeled split with class filter
    python train_stl10.py --split train+test --class_filter cat --network ncsnpp --device cuda:0
"""

import sys
sys.path.append("..")

import os
import functools
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from networks import ScoreNetUNet, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from stl10_datamodule import STL10GrayDataModule


def parse_class_filter(s):
    """Parse class filter string: 'cat', 'cat,dog', or 'all'/None."""
    if s is None or s.lower() == 'all' or s.lower() == 'none':
        return None
    parts = [p.strip() for p in s.split(',')]
    return parts if len(parts) > 1 else parts[0]


def class_filter_name(class_filter):
    if class_filter is None:
        return 'all'
    if isinstance(class_filter, list):
        return '_'.join(str(c) for c in class_filter)
    return str(class_filter)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on STL-10 grayscale (64x64)")
    parser.add_argument("--split", type=str, default="unlabeled",
                        choices=["unlabeled", "train", "test", "train+test"],
                        help="Dataset split (unlabeled=100k, train+test=13k with class filter)")
    parser.add_argument("--class_filter", type=str, default=None,
                        help="Class filter (only for labeled splits): 'cat', 'dog', 'cat,dog', or None")
    parser.add_argument("--sigma", type=float, default=25.0, help="Noise scale")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--ema_start", type=int, default=0, help="Start EMA after this epoch")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["unet", "ncsnpp"],
                        help="Network architecture: unet | ncsnpp")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--ckpt_every", type=int, default=50,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    L = 64  # STL-10 resized to 64x64

    # Parse class filter
    class_filter = parse_class_filter(args.class_filter)
    class_name = class_filter_name(class_filter)
    split_label = args.split.replace('+', '')  # "train+test" -> "traintest"

    # Data module (loads preprocessed .npy cache from data/)
    data_module = STL10GrayDataModule(
        data_dir='./data',
        batch_size=args.batch_size,
        normalize=True,
        num_workers=4,
        split=args.split,
        class_filter=class_filter,
    )
    data_module.prepare_data()
    data_module.setup()
    print(f"Training samples: {len(data_module.train_data)}")

    # Output directory
    output_dir = f"stl10_{split_label}_{class_name}_{args.network}"
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    # Save training data as (Lx, Ly, N) for correlation analysis
    train_images = data_module.train_data.tensors[0]  # (N, 1, 64, 64)
    train_cfgs = train_images[:, 0].numpy().transpose(1, 2, 0)  # (64, 64, N)
    train_data_path = f"{output_dir}/data/stl10_{class_name}_train_{L}x{L}.npy"
    np.save(train_data_path, train_cfgs)
    print(f"Training data saved: {train_data_path}, shape={train_cfgs.shape}")

    # Create score model
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn, periodic=False)
        print("Using NCSNpp2D (non-periodic)")
    else:
        score_model = ScoreNetUNet(marginal_prob_std_fn, image_size=L)
        print(f"Using ScoreNetUNet ({L}x{L})")

    score_model = torch.compile(score_model, mode="reduce-overhead")

    # Create diffusion model
    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=L,
        ema_start_epoch=args.ema_start,
        norm_min=-1.0,
        norm_max=1.0,
    )
    print(f"EMA starts at epoch: {args.ema_start}")

    # Callbacks
    checkpoint_periodic = ModelCheckpoint(
        dirpath=f"{output_dir}/models",
        filename="epoch={epoch:04d}",
        every_n_epochs=args.ckpt_every,
        save_top_k=-1,
    )

    # Trainer
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_periodic],
        default_root_dir=output_dir,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
