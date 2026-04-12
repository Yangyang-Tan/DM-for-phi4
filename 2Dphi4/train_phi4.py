"""
Train a score-based diffusion model on 2D phi4 lattice field configurations.

Usage:
    python train_phi4.py
    python train_phi4.py --epochs 100 --batch_size 128
"""

import sys
sys.path.append("..")

import functools
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from networks import ScoreNet, ScoreNetUNetPeriodic, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import FieldDataModule


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on 2D phi4")
    parser.add_argument("--L", type=int, default=128, help="Lattice size")
    parser.add_argument("--k", type=float, default=0.5, help="Hopping parameter")
    parser.add_argument("--l", type=float, default=0.022, help="Coupling constant")
    parser.add_argument("--sigma", type=float, default=150.0, help="Noise scale")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--ema_start", type=int, default=0, help="Start EMA after this epoch")
    parser.add_argument("--device", type=str, default="cuda:3", help="GPU device")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["scorenet", "unet", "ncsnpp"],
                        help="Network architecture: scorenet | unet | ncsnpp")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming training")
    args = parser.parse_args()

    # Default data path
    if args.data_path is None:
        # args.data_path = f"../data/cfgs_k={args.k}_l={args.l}_{args.L}^2_t=10.jld2"
        args.data_path = f"trainingdata/cfgs_langevin_k={args.k}_l={args.l}_{args.L}^2.jld2"

    # Create data module and setup to get normalization parameters
    data_module = FieldDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        normalize=True
    )
    data_module.setup()  # Compute cfgs_min, cfgs_max

    # Create marginal_prob_std function
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    # Create score model based on network choice
    if args.network == "unet":
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)
        print("Using UNet with downsampling (periodic BC)")
    elif args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn)
        print("Using NCSNpp2D (NCSN++ style, periodic BC)")
    else:
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
        print("Using ScoreNet (no downsampling)")

    score_model = torch.compile(score_model, mode="reduce-overhead")

    # Create diffusion model with normalization parameters
    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=args.L,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )
    print(f"Normalization range: [{data_module.cfgs_min:.4f}, {data_module.cfgs_max:.4f}]")
    print(f"EMA starts at epoch: {args.ema_start}")

    # Output directory for logs and models
    output_dir = f"phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}"
    print(f"Output directory: {output_dir}/")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/models",
        filename="{epoch:02d}-{train_loss_epoch:.4f}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=5,
    )

    checkpoint_callback2 = ModelCheckpoint(
        dirpath=f"{output_dir}/models",
        filename="epoch={epoch:04d}",
        every_n_epochs=50,
        # every_n_train_steps=1,
        save_top_k=-1,
    )
    # Trainer
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_callback2],
        default_root_dir=output_dir,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print(f"\nTraining complete! Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
