"""
Cross-L sampling: load a checkpoint trained on L_train and generate at L_sample.

Architecture is fully convolutional with circular padding, so the network can
process any L divisible by 8. Physics is *not* guaranteed: norm_min/max, sigma,
and the score itself were tuned on L_train, so results on L_sample are an
out-of-distribution test, not a bona fide L_sample model.

Usage:
    python sample_phi4_crossL.py --L_train 32 --L_sample 64 --k 0.26   --ep 4999
    python sample_phi4_crossL.py --L_train 32 --L_sample 64 --k 0.2705 --ep 9999
    python sample_phi4_crossL.py --L_train 32 --L_sample 64 --k 0.28   --ep 4999
"""

import sys
sys.path.append("..")

import os
import functools
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import NCSNpp2D, ScoreNet, ScoreNetUNetPeriodic
from diffusion_lightning import DiffusionModel, marginal_prob_std


def phi4_action(phi, k, l, phi_min, phi_max):
    """2D phi^4 action on normalised fields: phi in [-1,1]."""
    p = (phi[:, 0, :, :] + 1) / 2 * (phi_max - phi_min) + phi_min
    nb = torch.roll(p, 1, dims=1) + torch.roll(p, 1, dims=2)
    return torch.sum(-2 * k * p * nb + (1 - 2 * l) * p**2 + l * p**4, dim=(1, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L_train", type=int, default=32)
    p.add_argument("--L_sample", type=int, default=64)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--ep", type=int, required=True,
                   help="Epoch number to load. Resolves to "
                        "{run_dir}/models/epoch={ep:04d}.ckpt by exact match.")
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--method", type=str, default="em", choices=["em", "ode", "mala"])
    p.add_argument("--ode_method", type=str, default="dpm2")
    p.add_argument("--schedule", type=str, default="log", choices=["log", "linear", "quadratic", "cosine"])
    p.add_argument("--n_repeats", type=int, default=2)
    p.add_argument("--t_mh", type=float, default=0.01,
                   help="MALA-only: time at which to switch from SDE to MALA")
    p.add_argument("--mh_steps", type=int, default=200,
                   help="MALA-only: number of MALA steps in Phase 2")
    p.add_argument("--norm_for_action", type=str, default="train",
                   choices=["train", "ref"],
                   help="MALA-only: use L_train or L_sample HMC norm range when "
                        "computing the physical action.")
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_grid", type=int, default=4)
    p.add_argument("--run_dir", type=str, default=None,
                   help="Override the training run directory. By default it is "
                        "constructed as runs/phi4_L{L_train}_k{k}_l{l}_{network}; pass "
                        "this to load multi-L runs e.g. "
                        "runs/phi4_Lmulti8-16-32_k0.2705_l0.022_ncsnpp")
    p.add_argument("--tag", type=str, default=None,
                   help="Override output tag. Default encodes L_train; multi-L "
                        "users may want a custom tag.")
    args = p.parse_args()

    assert args.L_sample % 8 == 0, "L_sample must be divisible by 8 (3 stride-2 downs)"

    train_dir = (args.run_dir
                 or f"runs/phi4_L{args.L_train}_k{args.k}_l{args.l}_{args.network}")
    ckpt_path = Path(f"{train_dir}/models") / f"epoch={args.ep:04d}.ckpt"
    if not ckpt_path.exists():
        available = sorted(p.name for p in Path(f"{train_dir}/models").glob("epoch=*.ckpt"))
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            f"Available ({len(available)}): "
            f"{available[:3]} ... {available[-3:]}"
        )
    ckpt_path = str(ckpt_path)
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    sigma = h.get("sigma", 20.0)
    norm_min = h.get("norm_min", -4.0)
    norm_max = h.get("norm_max",  4.0)
    L_train_ckpt = h.get("L", args.L_train)
    # Auto-detect l_cond from state_dict (l_cond flag not stored in hparams,
    # but the L_embed submodule is when training used --l_cond).
    sd = ckpt.get("state_dict", {})
    l_cond_detected = any("L_embed" in k for k in sd.keys())
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]  "
          f"L_train(ckpt)={L_train_ckpt}  l_cond={l_cond_detected}")
    print(f"Sampling at L={args.L_sample}  (cross-L, network is fully convolutional)")

    # Build model and load weights
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn, l_cond=l_cond_detected)
    elif args.network == "scorenet":
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)
    model = DiffusionModel.load_from_checkpoint(ckpt_path, score_model=score_model)
    # CRITICAL override: tell DiffusionModel.sample() to use L_sample, not L_train.
    model.L = args.L_sample
    model = model.to(args.device).eval()

    # Output dir under the L_train run dir, but tagged with L_sample
    out_root = Path(train_dir) / "data_crossL"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.tag is not None:
        tag = args.tag
    else:
        tag = f"crossL_train{args.L_train}_sample{args.L_sample}_{args.method}_{args.schedule}_steps{args.num_steps}_ep{args.ep}"
        if args.method == "mala":
            tag += f"_tmh{args.t_mh}_mh{args.mh_steps}_norm{args.norm_for_action}"

    # Sample
    print(f"Sampling [{args.method}/{args.schedule}] num_steps={args.num_steps} n_repeats={args.n_repeats} num/rep={args.num_samples}  seed={args.seed}")
    reps = []
    for i in range(args.n_repeats):
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        if args.method == "em":
            reps.append(model.sample(args.num_samples, args.num_steps, schedule=args.schedule))
        elif args.method == "ode":
            reps.append(model.sample_ode(args.num_samples, args.num_steps,
                                         schedule=args.schedule, method=args.ode_method))
        else:  # mala
            # action_fn must use the *physical* phi^4 action; phi_min/max set
            # the normalised<->physical conversion. With --norm_for_action ref
            # we read L_sample HMC range from the reference jld2 file.
            if args.norm_for_action == "ref":
                import h5py
                ref_path = (Path("trainingdata") /
                            f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L_sample}^2.jld2")
                with h5py.File(str(ref_path), "r") as f:
                    cfgs_ref = np.array(f["cfgs"]).astype(np.float64)
                sa = int(np.argmax(cfgs_ref.shape))
                if sa != 0:
                    cfgs_ref = np.moveaxis(cfgs_ref, sa, 0)
                pmin, pmax = float(cfgs_ref.min()), float(cfgs_ref.max())
                print(f"  action norm: ref L={args.L_sample} → [{pmin:.4f},{pmax:.4f}]")
            else:
                pmin, pmax = norm_min, norm_max
                print(f"  action norm: train L={args.L_train} → [{pmin:.4f},{pmax:.4f}]")

            action_fn = functools.partial(phi4_action,
                                          k=args.k, l=args.l,
                                          phi_min=pmin, phi_max=pmax)
            samp_i, acc = model.sample_mala(
                args.num_samples, args.num_steps, args.t_mh, args.mh_steps,
                action_fn=action_fn, schedule=args.schedule,
            )
            print(f"  MALA acceptance rate: mean={acc.mean().item():.4f}  "
                  f"min={acc.min().item():.4f}  max={acc.max().item():.4f}")
            reps.append(samp_i)
    samples = torch.cat(reps, dim=0)
    samples_norm = samples[:, 0].cpu().numpy()  # (N, L, L) in [-1,1]

    # Denormalise back to physical phi (use training norms — that is the inverse map)
    samples_phys = (samples_norm + 1) / 2 * (norm_max - norm_min) + norm_min

    out_npy = out_root / f"samples_{tag}.npy"
    np.save(out_npy, samples_phys.transpose(1, 2, 0))  # (L, L, N), matches sample_phi4.py convention
    print(f"Saved: {out_npy}  shape={samples_phys.transpose(1,2,0).shape}")

    # Quick visual: 4x4 grid
    n = args.plot_grid
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples_phys[i], cmap="viridis")
        ax.axis("off")
    fig.suptitle(f"L_train={args.L_train} → L_sample={args.L_sample}, k={args.k}, λ={args.l}, sched={args.schedule}, steps={args.num_steps}", y=1.02)
    plt.tight_layout()
    out_png = out_root / f"samples_{tag}.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
