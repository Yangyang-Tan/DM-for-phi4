"""
Train a 2D phi^4 score-based diffusion model on configurations from MULTIPLE
lattice sizes simultaneously.

The whole point of this is to give the score function exposure to several
correlation-length-to-volume ratios so it can be evaluated on a held-out L
(e.g. train on {8,16,32}, test on 64) without IR-mode collapse.

The architecture is unchanged — NCSNpp2D / ScoreNet / ScoreNetUNetPeriodic
are fully convolutional with circular padding and so accept any L. The only
change is the data pipeline ([MultiLFieldDataModule] in ../data.py): pooled
[-1,1] normalisation, per-step random-L batching.

Usage:
    python train_phi4_multiL.py --L_list 8,16,32 --k 0.2705 --l 0.022 \
        --network ncsnpp --device cuda:0 --epochs 5000 --sigma 20

Sampling at any L afterwards uses sample_phi4_crossL.py with --L_train set
to the run's tag (the run-dir name encodes the L-list).
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

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from networks import ScoreNet, ScoreNetUNetPeriodic, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from data import MultiLFieldDataModule


class LogScaleCheckpoint(Callback):
    """Save checkpoints on a log-scale schedule (matches train_phi4.py)."""

    def __init__(self, dirpath, max_epochs, num_checkpoints=100):
        super().__init__()
        self.dirpath = dirpath
        log_epochs = np.unique(np.geomspace(1, max_epochs, num=num_checkpoints).astype(int))
        self.save_epochs = set(log_epochs.tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch in self.save_epochs:
            filepath = os.path.join(self.dirpath, f"epoch={epoch:04d}.ckpt")
            trainer.save_checkpoint(filepath)


def parse_L_list(s: str):
    return sorted(int(x) for x in s.split(",") if x.strip())


def main():
    p = argparse.ArgumentParser(description="Multi-L training for 2D phi^4 diffusion")
    p.add_argument("--L_list", type=str, default="8,16,32",
                   help="Comma-separated list of training L's (e.g. '8,16,32')")
    p.add_argument("--k", type=float, default=0.2705)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--sigma", type=float, default=20.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_prefix", type=str,
                   default="trainingdata/cfgs_wolff_fahmc",
                   help="Path prefix; full path is {prefix}_k={k}_l={l}_{L}^2.jld2")
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--num_ckpts", type=int, default=100)
    p.add_argument("--gpu_data", action="store_true",
                   help="Pin all training tensors on GPU to avoid H2D copies")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile (recommended for multi-L due to "
                        "shape-specific recompiles between batches)")
    p.add_argument("--l_cond", action="store_true",
                   help="Enable lattice-size conditioning in NCSNpp2D "
                        "(adds Gaussian Fourier embedding of 1/L to time "
                        "embedding). Auto-suffixes the output dir with _lcond.")
    p.add_argument("--output_suffix", type=str, default="")
    args = p.parse_args()

    L_list = parse_L_list(args.L_list)
    print(f"Training on L_list = {L_list}")

    data_paths = [
        f"{args.data_prefix}_k={args.k}_l={args.l}_{L}^2.jld2"
        for L in L_list
    ]
    for p_ in data_paths:
        if not os.path.isfile(p_):
            raise FileNotFoundError(p_)

    data_module = MultiLFieldDataModule(
        data_paths=data_paths,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device if args.gpu_data else None,
    )
    data_module.setup()

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn, l_cond=args.l_cond)
        print(f"Using NCSNpp2D (periodic BC, l_cond={args.l_cond})")
    elif args.network == "unet":
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)
        print("Using ScoreNetUNetPeriodic")
    else:
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
        print("Using ScoreNet (no downsampling, periodic)")

    if not args.no_compile:
        # dynamic=False: separate static-shape graph per L (3 graphs total for
        # L_list=[8,16,32]). Dynamic-shape compile chokes on the lattice
        # arithmetic (sympy hangs on pow_by_natural simplifications), so we
        # accept the per-shape compile cost in exchange for a smooth steady
        # state.
        score_model = torch.compile(score_model, mode="reduce-overhead",
                                    dynamic=False)
        print("torch.compile enabled (static, per-L specialization)")

    # Use the largest training L for self.L (only matters for sampling default
    # — sampling can override via model.L = L_target).
    L_for_state = max(L_list)
    model = DiffusionModel(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=L_for_state,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )

    L_tag = "-".join(str(L) for L in L_list)
    auto_suffix = "_lcond" if args.l_cond else ""
    output_dir = (f"runs/phi4_Lmulti{L_tag}_k{args.k}_l{args.l}_{args.network}"
                  f"{auto_suffix}{args.output_suffix}")
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    checkpoint_log = LogScaleCheckpoint(
        dirpath=f"{output_dir}/models",
        max_epochs=args.epochs,
        num_checkpoints=args.num_ckpts,
    )
    print(f"Log-scale checkpoints ({len(checkpoint_log.save_epochs)} total): "
          f"{sorted(checkpoint_log.save_epochs)[:5]} ... "
          f"{sorted(checkpoint_log.save_epochs)[-5:]}")

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

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
