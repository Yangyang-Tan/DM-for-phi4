"""
Train a score-based diffusion model on ChestMNIST.

Usage:
    python train_chestmnist.py
    python train_chestmnist.py --epochs 1000 --batch_size 128
"""

import sys
sys.path.append("..")

import functools
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from networks import ScoreNetUNet
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import ChestMNISTDataModule


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on ChestMNIST")
    parser.add_argument("--sigma", type=float, default=55.0, help="Noise scale")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda:2", help="GPU device")
    parser.add_argument("--size", type=int, default=64, help="Image size")
    args = parser.parse_args()

    # Create marginal_prob_std function
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    # Create score model (U-Net for 64x64 images)
    score_model = ScoreNetUNet(marginal_prob_std_fn, image_size=args.size)

    # Create diffusion model
    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=args.size,  # ChestMNIST is 64x64
    )

    # Create data module (data stays on GPU to avoid transfer latency)
    data_module = ChestMNISTDataModule(
        batch_size=args.batch_size, 
        size=args.size,
        device=args.device
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/chestmnist",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=3,
        filename="chestmnist-{epoch:02d}-{train_loss_epoch:.4f}",
    )

    # Trainer
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
