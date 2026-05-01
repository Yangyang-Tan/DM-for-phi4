"""
In-process multi-epoch sweep sampler for 3D phi^4.

Mirrors sample_phi4_sweep.py (2D) but for DiffusionModel3D + NCSNpp3D and 5D
tensor shapes (N, 1, L, L, L). Compiles once, runs over an epoch list,
loading each ckpt's state_dict into the shared compiled module.

One process handles ONE method (em OR ode). The caller launches two processes:
  cuda:2 -> em (SDE, bf16 autocast inside dm.sample)
  cuda:3 -> ode (fp32, no autocast)
so both GPUs run in parallel.

Output format matches sample_phi4.py:
    {run_dir}/data/samples_{method}_steps{N}_epoch={NNNN}.npy
with shape (L, L, L, total_samples) renormalized to [norm_min, norm_max].

Usage:
  python sample_phi4_3d_sweep.py --k 0.2 --l 0.9 --sigma 2760 \
         --method em  --device cuda:2 --num_samples 256 --n_repeats 4
  python sample_phi4_3d_sweep.py --k 0.2 --l 0.9 --sigma 2760 \
         --method ode --device cuda:3 --num_samples 256 --n_repeats 4
"""
import sys
sys.path.append("..")

import argparse
import functools
import os
import time

import numpy as np
import torch

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std


DEFAULT_EPOCHS = [
    "0001", "0002", "0003", "0005", "0009", "0016", "0028", "0045", "0079",
    "0138", "0242", "0422", "0739", "1291", "2257", "3593", "6280", "10000",
]


def save_samples(samples_norm_tensor, method, steps, ep, out_dir,
                 norm_min, norm_max):
    """samples_norm_tensor: (N, 1, L, L, L) in [-1,1]; save as (L,L,L,N)."""
    x = samples_norm_tensor[:, 0].cpu().numpy()  # (N, L, L, L)
    x = (x + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
    x = x.transpose(1, 2, 3, 0)                  # (L, L, L, N)
    out = f"{out_dir}/samples_{method}_steps{steps}_epoch={ep}.npy"
    np.save(out, x)
    return out, x.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, required=True)
    p.add_argument("--network", type=str, default="ncsnpp")
    p.add_argument("--sigma", type=float, default=None,
                   help="If set, output_suffix becomes '_sigma{sigma}'.")
    p.add_argument("--output_suffix", type=str, default="")
    p.add_argument("--epochs", type=str, default=None,
                   help="Comma-separated; default 18 log-spaced.")
    p.add_argument("--method", type=str, required=True,
                   choices=["em", "ode"],
                   help="em = SDE Euler-Maruyama (bf16); ode = DPM-2 ODE (fp32).")
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--sde_steps", type=int, default=2000)
    p.add_argument("--ode_steps", type=int, default=400)
    p.add_argument("--ode_method", type=str, default="dpm2")
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--schedule", type=str, default="log")
    args = p.parse_args()

    if args.sigma is not None:
        args.output_suffix = f"_sigma{int(args.sigma)}"

    epochs = (args.epochs.split(",") if args.epochs else DEFAULT_EPOCHS)
    epochs = [e.strip() for e in epochs if e.strip()]

    run_dir = (f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_"
               f"{args.network}{args.output_suffix}")
    models_dir = f"{run_dir}/models"
    out_dir = f"{run_dir}/data"
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    torch.set_float32_matmul_precision("medium")

    # ── read first ckpt for hparams ─────────────────────────────────────
    first_ckpt = f"{models_dir}/epoch={epochs[0]}.ckpt"
    if not os.path.isfile(first_ckpt):
        sys.exit(f"[sweep3d] ERROR: first ckpt missing: {first_ckpt}")
    ck = torch.load(first_ckpt, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {})
    sigma = float(hp.get("sigma", 150.0))
    L_ckpt = int(hp.get("L", args.L))
    norm_min = float(hp.get("norm_min"))
    norm_max = float(hp.get("norm_max"))
    total_samples = args.num_samples * args.n_repeats
    steps = args.sde_steps if args.method == "em" else args.ode_steps
    print(f"[sweep3d] run_dir={run_dir}")
    print(f"[sweep3d] hparams: sigma={sigma}  L={L_ckpt}  norm=[{norm_min:.4f}, {norm_max:.4f}]")
    print(f"[sweep3d] epochs (n={len(epochs)}): {epochs}")
    print(f"[sweep3d] method={args.method}  steps={steps}  "
          f"batch={args.num_samples}×{args.n_repeats}={total_samples}  seed={args.seed}")
    del ck

    # ── build compiled score model once ─────────────────────────────────
    mps = functools.partial(marginal_prob_std, sigma=sigma)
    raw = NCSNpp3D(mps)
    compiled_score = torch.compile(raw, mode=args.compile_mode)
    print(f"[sweep3d] NCSNpp3D compiled (mode={args.compile_mode})")

    # ── loop over epochs ────────────────────────────────────────────────
    for ep_idx, ep in enumerate(epochs):
        ckpt_path = f"{models_dir}/epoch={ep}.ckpt"
        if not os.path.isfile(ckpt_path):
            print(f"[sweep3d] SKIP missing: {ckpt_path}")
            continue

        t0 = time.time()
        dm = DiffusionModel3D.load_from_checkpoint(
            ckpt_path, score_model=compiled_score
        )
        dm = dm.to(device).eval()

        reps = []
        t_s = time.time()
        for i in range(args.n_repeats):
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            if args.method == "em":
                reps.append(dm.sample(args.num_samples, args.sde_steps,
                                      schedule=args.schedule))
            else:
                reps.append(dm.sample_ode(args.num_samples, args.ode_steps,
                                          schedule=args.schedule,
                                          method=args.ode_method))
        samples = torch.cat(reps, dim=0)
        t_dur = time.time() - t_s
        nfe = steps * args.n_repeats
        rate = nfe / t_dur
        path, shape = save_samples(samples, args.method, steps, ep,
                                   out_dir, norm_min, norm_max)
        print(f"[sweep3d] ep={ep} {args.method} {t_dur:6.1f}s  "
              f"{rate:5.2f} it/s  shape={shape}  "
              f"-> {os.path.basename(path)}  ({ep_idx+1}/{len(epochs)})")
        del samples, reps, dm
        torch.cuda.empty_cache()

    print(f"[sweep3d] DONE: all {len(epochs)} epochs processed")


if __name__ == "__main__":
    main()
