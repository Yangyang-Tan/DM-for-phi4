"""
3D cross-L sampling: load a checkpoint trained on L_train and generate at
L_sample. Mirrors 2Dphi4/sample_phi4_crossL.py.

NCSNpp3D is fully convolutional with circular padding so accepts any L
divisible by 8. Physics is *not* guaranteed at L_sample > L_train_max.
"""

import sys
import os
import functools
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D, ScoreNet3D, ScoreNet3DUNetPeriodic
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L_train", type=int, default=32)
    p.add_argument("--L_sample", type=int, default=64)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--ep", type=int, required=True,
                   help="Epoch number; resolves to {run_dir}/models/epoch={ep:04d}.ckpt")
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--method", type=str, default="em", choices=["em", "ode"])
    p.add_argument("--ode_method", type=str, default="dpm2")
    p.add_argument("--schedule", type=str, default="log",
                   choices=["log", "linear", "quadratic", "cosine"])
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_dir", type=str, default=None,
                   help="Override training run dir. Default: "
                        "runs/phi4_3d_L{L_train}_k{k}_l{l}_{network}")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    assert args.L_sample % 8 == 0, "L_sample must be divisible by 8"

    train_dir = (args.run_dir or
                 f"runs/phi4_3d_L{args.L_train}_k{args.k}_l{args.l}_{args.network}")
    ckpt_path = Path(f"{train_dir}/models") / f"epoch={args.ep:04d}.ckpt"
    if not ckpt_path.exists():
        available = sorted(p.name for p in Path(f"{train_dir}/models").glob("epoch=*.ckpt"))
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            f"Available ({len(available)}): {available[:3]} ... {available[-3:]}"
        )
    ckpt_path = str(ckpt_path)
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    sigma = h.get("sigma", 150.0)
    norm_min = h.get("norm_min", -6.0)
    norm_max = h.get("norm_max", 6.0)
    L_train_ckpt = h.get("L", args.L_train)
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]  "
          f"L_train(ckpt)={L_train_ckpt}")
    print(f"Sampling at L={args.L_sample}  (cross-L, network is fully convolutional)")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp3D(marginal_prob_std_fn)
    elif args.network == "scorenet":
        score_model = ScoreNet3D(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNet3DUNetPeriodic(marginal_prob_std_fn)
    model = DiffusionModel3D.load_from_checkpoint(ckpt_path, score_model=score_model)
    model.L = args.L_sample
    model = model.to(args.device).eval()

    out_root = Path(train_dir) / "data_crossL"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.tag is not None:
        tag = args.tag
    else:
        tag = (f"crossL_train{args.L_train}_sample{args.L_sample}_"
               f"{args.method}_{args.schedule}_steps{args.num_steps}_ep{args.ep}")

    print(f"Sampling [{args.method}/{args.schedule}] num_steps={args.num_steps} "
          f"n_repeats={args.n_repeats} num/rep={args.num_samples}  seed={args.seed}")
    reps = []
    for i in range(args.n_repeats):
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        if args.method == "em":
            reps.append(model.sample(args.num_samples, args.num_steps,
                                     schedule=args.schedule))
        else:
            reps.append(model.sample_ode(args.num_samples, args.num_steps,
                                         schedule=args.schedule, method=args.ode_method))
    samples = torch.cat(reps, dim=0)
    samples_norm = samples[:, 0].cpu().numpy()  # (N, L, L, L) in [-1,1]

    samples_phys = (samples_norm + 1) / 2 * (norm_max - norm_min) + norm_min

    out_npy = out_root / f"samples_{tag}.npy"
    # Save as (L, L, L, N) to match 3D conventions
    np.save(out_npy, samples_phys.transpose(1, 2, 3, 0))
    print(f"Saved: {out_npy}  shape={samples_phys.transpose(1,2,3,0).shape}")

    # Quick visual: slice through z=L/2, 4x4 grid
    n = 4
    z_mid = args.L_sample // 2
    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples_phys[i, :, :, z_mid], cmap="viridis")
        ax.axis("off")
    fig.suptitle(f"L_train={args.L_train} → L_sample={args.L_sample} "
                 f"(z={z_mid} slice), k={args.k}, λ={args.l}", y=1.02)
    plt.tight_layout()
    out_png = out_root / f"samples_{tag}.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
