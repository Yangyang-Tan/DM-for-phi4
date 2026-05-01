"""
Train a score-based diffusion model on CelebA grayscale images.

Supports 64x64 and 128x128 resolutions.

Usage:
    python train_celeba.py --image_size 64 --network ncsnpp --device cuda:0
    python train_celeba.py --image_size 128 --network ncsnpp --batch_size 64 --device cuda:2
"""

import sys
sys.path.append("..")

import os
import functools
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from networks import ScoreNetUNet, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from celeba_datamodule import CelebAGrayDataModule


class LogScaleCheckpoint(Callback):
    """Save checkpoints on a log-scale schedule: more frequent early, sparser later."""

    def __init__(self, dirpath, max_epochs, num_checkpoints=50):
        super().__init__()
        self.dirpath = dirpath
        # Generate log-spaced epoch indices
        log_epochs = np.unique(np.geomspace(1, max_epochs, num=num_checkpoints).astype(int))
        self.save_epochs = set(log_epochs.tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-based
        if epoch in self.save_epochs:
            filepath = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            trainer.save_checkpoint(filepath)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on CelebA grayscale")
    parser.add_argument("--image_size", type=int, default=64, choices=[64, 128],
                        help="Image size: 64 or 128")
    parser.add_argument("--sigma", type=float, default=25.0, help="Noise scale")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--ema_start", type=int, default=0, help="Start EMA after this epoch")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["unet", "ncsnpp"],
                        help="Network architecture: unet | ncsnpp")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--num_ckpts", type=int, default=100,
                        help="Total number of checkpoints (log-spaced)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (None = all ~200k)")
    parser.add_argument("--gpu_data", action="store_true",
                        help="Load all data onto GPU (saves CPU-GPU transfer, needs ~3GB for 64x64)")
    args = parser.parse_args()

    L = args.image_size

    # Data module
    data_module = CelebAGrayDataModule(
        data_dir='./data',
        image_size=L,
        batch_size=args.batch_size,
        normalize=True,
        num_workers=1,
        max_samples=args.max_samples,
        device=args.device if args.gpu_data else None,
    )
    data_module.prepare_data()
    data_module.setup()
    print(f"Training samples: {len(data_module.train_data)}")

    # Output directory
    output_dir = f"runs/celeba_{L}_{args.network}"
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    # Save training data as (Lx, Ly, N) for correlation analysis
    # Use a subset to avoid huge files for 200k images
    train_images = data_module.train_data.tensors[0]  # (N, 1, L, L)
    n_save = min(len(train_images), 50000)
    train_cfgs = train_images[:n_save, 0].numpy().transpose(1, 2, 0)  # (L, L, N)
    train_data_path = f"{output_dir}/data/celeba_train_{L}x{L}.npy"
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

    # Callbacks — log-scale checkpoint schedule
    checkpoint_log = LogScaleCheckpoint(
        dirpath=f"{output_dir}/models",
        max_epochs=args.epochs,
        num_checkpoints=args.num_ckpts,
    )
    print(f"Log-scale checkpoints at epochs: {sorted(checkpoint_log.save_epochs)}")

    # Trainer
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_log],
        default_root_dir=output_dir,
        precision="bf16-mixed",
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
