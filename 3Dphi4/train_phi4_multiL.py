"""
Train a 3D phi^4 score-based diffusion model on configurations from MULTIPLE
lattice sizes simultaneously (L_list, e.g. {4,8,16,32}).

Mirrors 2Dphi4/train_phi4_multiL.py. NCSNpp3D is fully convolutional with
circular padding so accepts any L; per-step single-L batches via
MultiLBatchSampler.

Usage:
    python train_phi4_multiL.py --L_list 4,8,16,32 --k 0.1923 --l 0.9 \\
        --network ncsnpp --device cuda:0 --epochs 10000 --sigma 375
"""

import os
import functools
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from networks_3d import ScoreNet3D, ScoreNet3DUNetPeriodic, NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from data_3d import MultiLFieldDataModule3D


class LogScaleCheckpoint(Callback):
    """Save checkpoints on a log-scale schedule."""

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
    p = argparse.ArgumentParser(description="Multi-L training for 3D phi^4 diffusion")
    p.add_argument("--L_list", type=str, default="4,8,16,32",
                   help="Comma-separated training L's (e.g. '4,8,16,32')")
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--sigma", type=float, default=375.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--ema_start", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_prefix", type=str,
                   default="trainingdata/cfgs_wolff_fahmc",
                   help="Path prefix; full path is {prefix}_k={k}_l={l}_{L}^3.jld2")
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--num_ckpts", type=int, default=100)
    p.add_argument("--gpu_data", action="store_true",
                   help="Pin all training tensors on GPU to avoid H2D copies")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile (recommended for multi-L due to "
                        "shape-specific recompiles between batches)")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead",
                   choices=["reduce-overhead", "max-autotune"],
                   help="torch.compile mode. reduce-overhead: ~30-60s/shape "
                        "compile, CUDA Graphs only — best for short runs. "
                        "max-autotune: ~5-10min/shape compile, also Triton "
                        "kernel autotuning — 15-30%% steady-state speedup, "
                        "worth it on long fresh runs.")
    p.add_argument("--output_suffix", type=str, default="")
    args = p.parse_args()

    L_list = parse_L_list(args.L_list)
    print(f"Training on L_list = {L_list}")

    data_paths = [
        f"{args.data_prefix}_k={args.k}_l={args.l}_{L}^3.jld2"
        for L in L_list
    ]
    for p_ in data_paths:
        if not os.path.isfile(p_):
            raise FileNotFoundError(p_)

    data_module = MultiLFieldDataModule3D(
        data_paths=data_paths,
        batch_size=args.batch_size,
        normalize=True,
        device=args.device if args.gpu_data else None,
    )
    data_module.setup()

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    if args.network == "ncsnpp":
        score_model = NCSNpp3D(marginal_prob_std_fn)
        print("Using NCSNpp3D (periodic BC)")
    elif args.network == "unet":
        score_model = ScoreNet3DUNetPeriodic(marginal_prob_std_fn)
        print("Using ScoreNet3DUNetPeriodic")
    else:
        score_model = ScoreNet3D(marginal_prob_std_fn, periodic=True)
        print("Using ScoreNet3D (periodic, no downsampling)")

    if not args.no_compile:
        # dynamic=False: separate static-shape graph per L. Dynamic-shape
        # compile chokes on lattice arithmetic (sympy hangs on
        # pow_by_natural simplifications), as observed in 2D.
        score_model = torch.compile(score_model, mode=args.compile_mode,
                                    dynamic=False)
        print(f"torch.compile enabled (mode={args.compile_mode}, "
              f"static per-L specialization)")

    L_for_state = max(L_list)
    model = DiffusionModel3D(
        score_model=score_model,
        sigma=args.sigma,
        lr=args.lr,
        L=L_for_state,
        ema_start_epoch=args.ema_start,
        norm_min=data_module.cfgs_min,
        norm_max=data_module.cfgs_max,
    )

    L_tag = "-".join(str(L) for L in L_list)
    output_dir = (f"runs/phi4_3d_Lmulti{L_tag}_k{args.k}_l{args.l}_{args.network}"
                  f"{args.output_suffix}")
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    print(f"Output directory: {output_dir}/")

    # Save full training config (CLI args + derived) for reproducibility
    import yaml
    cfg = {**vars(args),
           "L_list": L_list,
           "norm_min": data_module.cfgs_min,
           "norm_max": data_module.cfgs_max,
           "n_total_cfgs": data_module.N_total,
           "param_count_M": sum(p.numel() for p in score_model.parameters()) / 1e6}
    with open(f"{output_dir}/training_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved training_config.yaml")

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
