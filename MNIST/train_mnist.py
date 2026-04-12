"""
Train a score-based diffusion model on MNIST.

Usage:
    python train_mnist.py
    python train_mnist.py --epochs 100 --batch_size 128
"""

import sys
sys.path.append("..")
import functools
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from networks import ScoreNetUNet
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import MNISTDataModule


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on MNIST")
    parser.add_argument("--sigma", type=float, default=25.0, help="Noise scale")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda:3", help="GPU device")
    args = parser.parse_args()

    # Create marginal_prob_std function
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    # Create score model (U-Net for MNIST 28x28)
    score_model = ScoreNetUNet(marginal_prob_std_fn)

    # Create diffusion model
    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=28,  # MNIST is 28x28
    )

    # Create data module (data stays on GPU to avoid transfer latency)
    data_module = MNISTDataModule(batch_size=args.batch_size, device=args.device)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/mnist",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=3,
        filename="mnist-{epoch:02d}-{train_loss_epoch:.4f}",
    )

    # Trainer (extract device index from "cuda:X")
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_callback],
    )

    # Train
    trainer.fit(model, data_module)

    print(f"\nTraining complete! Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
