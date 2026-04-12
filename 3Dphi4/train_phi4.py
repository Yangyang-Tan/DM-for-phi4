"""
Train a score-based diffusion model on 3D phi4 lattice field configurations.

Usage:
    python train_phi4.py
    python train_phi4.py --epochs 100 --batch_size 32
    python train_phi4.py --L 32 --k 0.5 --l 0.022 --data_path ../data/cfgs_k=0.5_l=0.022_32^3_t=10.jld2
"""

import re
import functools
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from networks_3d import ScoreNet3D, ScoreNet3DUNetPeriodic, NCSNpp3D, NCSNpp3DSimple
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from data_3d import FieldDataModule3D


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on 3D phi4")
    parser.add_argument("--L", type=int, default=32, help="Lattice size (L x L x L)")
    parser.add_argument("--k", type=float, default=0.5, help="Hopping parameter")
    parser.add_argument("--l", type=float, default=0.9, help="Coupling constant")
    parser.add_argument("--sigma", type=float, default=150.0, help="Noise scale")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (smaller for 3D due to memory)")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--ema_start", type=int, default=1000, help="Start EMA after this epoch")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--network", type=str, default="ncsnpp", 
                        choices=["scorenet", "unet", "ncsnpp", "ncsnpp_simple"],
                        help="Network architecture: scorenet (no downsampling), unet (with downsampling), "
                             "ncsnpp (full NCSN++), ncsnpp_simple (lightweight NCSN++)")
    args = parser.parse_args()

    # Default data path for 3D lattice
    if args.data_path is None:
        args.data_path = f"../data/cfgs_k={args.k}_l={args.l}_{args.L}^3_t=10.jld2"

    # Infer L, k, l from data filename (overrides command-line defaults)
    fname = args.data_path
    m_k = re.search(r'_k=([\d.]+)', fname)
    m_l = re.search(r'_l=([\d.]+)', fname)
    m_L = re.search(r'_(\d+)\^3', fname)
    if m_k:
        args.k = float(m_k.group(1))
    if m_l:
        args.l = float(m_l.group(1))
    if m_L:
        args.L = int(m_L.group(1))
    print(f"Inferred from data path: L={args.L}, k={args.k}, l={args.l}")

    # Create data module and setup to get normalization parameters
    data_module = FieldDataModule3D(
        data_path=args.data_path,
        batch_size=args.batch_size,
        normalize=True
    )
    data_module.setup()  # Compute cfgs_min, cfgs_max

    # Create marginal_prob_std function
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    # Create 3D score model based on network choice
    if args.network == "unet":
        score_model = ScoreNet3DUNetPeriodic(marginal_prob_std_fn)
        print("Using 3D UNet with downsampling (periodic BC)")
    elif args.network == "ncsnpp":
        score_model = NCSNpp3D(marginal_prob_std_fn)
        print("Using NCSNpp3D (NCSN++ with ResNet blocks + FiLM)")
    elif args.network == "ncsnpp_simple":
        # Alias, same as ncsnpp
        score_model = NCSNpp3DSimple(marginal_prob_std_fn)
        print("Using NCSNpp3D (NCSN++ with ResNet blocks + FiLM)")
    else:
        score_model = ScoreNet3D(marginal_prob_std_fn, periodic=True)
        print("Using ScoreNet3D (no downsampling)")

    score_model = torch.compile(score_model, mode="reduce-overhead")

    model = DiffusionModel3D(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=args.L,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )
    n_params = sum(p.numel() for p in score_model.parameters())
    print(f"Score model parameters: {n_params:,} ({n_params/1e6:.3f}M)")
    print(f"Normalization range: [{data_module.cfgs_min:.4f}, {data_module.cfgs_max:.4f}]")
    print(f"EMA starts at epoch: {args.ema_start}")
    print(f"Training 3D phi4 model with L={args.L}, k={args.k}, l={args.l}")

    # Output directory for logs and models
    output_dir = f"phi4_3d_L{args.L}_k{args.k}_l{args.l}_{args.network}"
    print(f"Output directory: {output_dir}/")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/models",
        filename="{epoch:02d}-{train_loss_epoch:.4f}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=2,
    )
    # Save checkpoint every 5 epochs
    periodic_checkpoint = ModelCheckpoint(
        dirpath=f"{output_dir}/models",
        filename="epoch={epoch:04d}",
        every_n_epochs=50,
        # every_n_train_steps=5,
        save_top_k=-1,  # Keep all periodic checkpoints
    )

    # Trainer (with gradient clipping to prevent explosion)
    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[device_id],
        callbacks=[checkpoint_callback, periodic_checkpoint],
        default_root_dir=output_dir,
        precision="bf16-mixed",
        # gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # Train (optionally resume from checkpoint)
    trainer.fit(model, data_module)
    # trainer.fit(model, data_module, ckpt_path="/home/tyy/DM/DMasSQ-main/3Dphi4/phi4_3d_L64_k0.5_l2.5_unet/models/epoch=301-train_loss_epoch=12164.2393.ckpt")
    print(f"\nTraining complete! Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
