"""
Train a score-based diffusion model on 2D phi4 lattice field configurations.

Usage:
    python train_phi4.py
    python train_phi4.py --epochs 100 --batch_size 128
"""

import sys
sys.path.append("..")

import os
import functools
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from networks import ScoreNet, ScoreNetUNetPeriodic, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import FieldDataModule


class LogScaleCheckpoint(Callback):
    """Save checkpoints on a log-scale schedule: denser early, sparser later."""

    def __init__(self, dirpath, max_epochs, num_checkpoints=100):
        super().__init__()
        self.dirpath = dirpath
        log_epochs = np.unique(np.geomspace(1, max_epochs, num=num_checkpoints).astype(int))
        self.save_epochs = set(log_epochs.tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-based
        if epoch in self.save_epochs:
            filepath = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            trainer.save_checkpoint(filepath)


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
    parser.add_argument("--num_ckpts", type=int, default=100,
                        help="Number of log-spaced checkpoints to save")
    parser.add_argument("--gpu_data", action="store_true",
                        help="Load all training data onto GPU once (avoids per-epoch H2D transfer)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to output directory name (e.g. '_sigma150')")
    args = parser.parse_args()

    # Default data path
    if args.data_path is None:
        # args.data_path = f"../data/cfgs_k={args.k}_l={args.l}_{args.L}^2_t=10.jld2"
        args.data_path = f"trainingdata/cfgs_langevin_k={args.k}_l={args.l}_{args.L}^2.jld2"

    # Create data module and setup to get normalization parameters
    data_module = FieldDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device if args.gpu_data else None,
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
    output_dir = f"phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}{args.output_suffix}"
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    # Log-scale checkpoint callback
    checkpoint_log = LogScaleCheckpoint(
        dirpath=f"{output_dir}/models",
        max_epochs=args.epochs,
        num_checkpoints=args.num_ckpts,
    )
    print(f"Log-scale checkpoints ({len(checkpoint_log.save_epochs)} total): "
          f"{sorted(checkpoint_log.save_epochs)[:5]} ... "
          f"{sorted(checkpoint_log.save_epochs)[-5:]}")

    # Trainer
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_log],
        default_root_dir=output_dir,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
