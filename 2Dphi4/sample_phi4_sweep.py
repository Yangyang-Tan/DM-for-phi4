"""
In-process multi-epoch sweep sampler for 2D phi^4.

Compiles the score model ONCE with torch.compile, then walks through an epoch
list, loading each ckpt's state_dict into the shared compiled module. This
avoids the 5-10min per-process compile cost that sample_phi4.py incurs when
called 18× in a shell loop.

Runs both em (SDE, bf16 via autocast inside diffusion_lightning) and
ode/dpm2 (fp32) at each epoch, with fixed seed across epochs so IC matches.

Output format matches sample_phi4.py:
    {run_dir}/data/samples_{em,ode}_steps{N}_epoch={NNNN}.npy
with shape (L, L, total_samples) renormalized to the training [norm_min, norm_max].

Usage:
    python sample_phi4_sweep.py --k 0.28   --sigma 640 --device cuda:2
    python sample_phi4_sweep.py --k 0.2705 --sigma 450 --device cuda:3

Epoch list defaults to the 18 log-spaced checkpoints in run_sweep_samples_L128.sh.
"""
import sys
sys.path.append("..")

import argparse
import functools
import os
import time

import numpy as np
import torch

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


DEFAULT_EPOCHS = [
    "0001", "0002", "0003", "0005", "0009", "0016", "0028", "0045", "0079",
    "0138", "0242", "0422", "0739", "1291", "2257", "3593", "6280", "10000",
]


def save_samples(samples_norm_tensor, method, steps, ep, out_dir,
                 norm_min, norm_max):
    """samples_norm_tensor: (N, 1, L, L) in [-1,1]; save as (L,L,N) in original range."""
    x = samples_norm_tensor[:, 0].cpu().numpy()
    x = (x + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
    x = x.transpose(1, 2, 0)
    out = f"{out_dir}/samples_{method}_steps{steps}_epoch={ep}.npy"
    np.save(out, x)
    return out, x.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--network", type=str, default="ncsnpp")
    p.add_argument("--sigma", type=float, default=None,
                   help="If set, adds output_suffix '_sigma{sigma}' (overrides --output_suffix).")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Suffix on run dir, e.g. '_sigma450'. Ignored if --sigma given.")
    p.add_argument("--epochs", type=str, default=None,
                   help="Comma-separated epoch numbers (e.g. 0001,0002,10000). "
                        "Default: 18 log-spaced epochs.")
    p.add_argument("--num_samples", type=int, default=1024,
                   help="Batch size per rep (total = num_samples × n_repeats)")
    p.add_argument("--n_repeats", type=int, default=2)
    p.add_argument("--sde_steps", type=int, default=2000)
    p.add_argument("--ode_steps", type=int, default=400)
    p.add_argument("--ode_method", type=str, default="dpm2",
                   choices=["dpm1", "dpm2", "dpm3"])
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

    run_dir = (f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_"
               f"{args.network}{args.output_suffix}")
    models_dir = f"{run_dir}/models"
    out_dir = f"{run_dir}/data"
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    torch.set_float32_matmul_precision("medium")

    # ── 1. read first ckpt for hparams (sigma, L, norm range) ────────────
    first_ckpt = f"{models_dir}/epoch={epochs[0]}.ckpt"
    if not os.path.isfile(first_ckpt):
        sys.exit(f"[sweep] ERROR: first ckpt missing: {first_ckpt}")
    ck = torch.load(first_ckpt, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {})
    sigma = float(hp.get("sigma", 150.0))
    L_ckpt = int(hp.get("L", args.L))
    norm_min = float(hp.get("norm_min"))
    norm_max = float(hp.get("norm_max"))
    print(f"[sweep] run_dir={run_dir}")
    print(f"[sweep] hparams: sigma={sigma}  L={L_ckpt}  norm=[{norm_min:.4f}, {norm_max:.4f}]")
    print(f"[sweep] epochs (n={len(epochs)}): {epochs}")
    print(f"[sweep] per call: em={args.sde_steps}steps  ode={args.ode_method}/{args.ode_steps}steps  "
          f"batch={args.num_samples}×{args.n_repeats}  seed={args.seed}")
    del ck

    # ── 2. build score model once, wrap with torch.compile ───────────────
    mps = functools.partial(marginal_prob_std, sigma=sigma)
    raw = NCSNpp2D(mps)
    compiled_score = torch.compile(raw, mode=args.compile_mode)
    print(f"[sweep] NCSNpp2D compiled (mode={args.compile_mode})")

    # ── 3. loop over epochs ──────────────────────────────────────────────
    for ep_idx, ep in enumerate(epochs):
        ckpt_path = f"{models_dir}/epoch={ep}.ckpt"
        if not os.path.isfile(ckpt_path):
            print(f"[sweep] SKIP missing: {ckpt_path}")
            continue

        t0 = time.time()
        # load_from_checkpoint reuses compiled_score via the score_model kwarg.
        # DiffusionModel.on_load_checkpoint bidirectionally adapts _orig_mod.
        dm = DiffusionModel.load_from_checkpoint(
            ckpt_path, score_model=compiled_score
        )
        dm = dm.to(device).eval()

        # ── em (bf16 autocast is applied inside dm.sample) ─────────────
        reps_em = []
        t_em_start = time.time()
        for i in range(args.n_repeats):
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            reps_em.append(dm.sample(args.num_samples, args.sde_steps,
                                     schedule=args.schedule))
        em_samples = torch.cat(reps_em, dim=0)
        t_em = time.time() - t_em_start
        nfe_em = args.sde_steps * args.n_repeats
        em_rate = nfe_em / t_em
        path_em, shape_em = save_samples(em_samples, "em", args.sde_steps,
                                         ep, out_dir, norm_min, norm_max)
        print(f"[sweep] ep={ep} em  {t_em:6.1f}s  {em_rate:5.2f} it/s  "
              f"shape={shape_em}  -> {os.path.basename(path_em)}")
        del em_samples, reps_em

        # ── ode/dpm2 (fp32, no autocast) ───────────────────────────────
        reps_ode = []
        t_ode_start = time.time()
        for i in range(args.n_repeats):
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            reps_ode.append(dm.sample_ode(args.num_samples, args.ode_steps,
                                          schedule=args.schedule,
                                          method=args.ode_method))
        ode_samples = torch.cat(reps_ode, dim=0)
        t_ode = time.time() - t_ode_start
        nfe_ode = args.ode_steps * args.n_repeats
        ode_rate = nfe_ode / t_ode
        path_ode, shape_ode = save_samples(ode_samples, "ode", args.ode_steps,
                                           ep, out_dir, norm_min, norm_max)
        print(f"[sweep] ep={ep} ode {t_ode:6.1f}s  {ode_rate:5.2f} it/s  "
              f"shape={shape_ode}  -> {os.path.basename(path_ode)}")
        del ode_samples, reps_ode, dm
        torch.cuda.empty_cache()

        t_total = time.time() - t0
        print(f"[sweep] ep={ep} total {t_total:.1f}s  "
              f"({ep_idx+1}/{len(epochs)})")

    print(f"[sweep] DONE: all {len(epochs)} epochs processed")


if __name__ == "__main__":
    main()
