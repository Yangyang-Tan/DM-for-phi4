"""
Multi-grid + MALA refinement for cross-L sampling.

Pipeline
--------
  Phase 1: sample at L_intermediate using a multi-L DM ckpt (in-distribution).
  Phase 2: bilinear-upsample the L_intermediate samples to L_target.
  Phase 3: MALA at L_target using the *true* phi^4 action gradient as the drift.

Why action-gradient MALA, not DM-score MALA, at L_target?
  The DM is OOD at L_target=128 (its score in the IR is unreliable). The true
  phi^4 action is exact at any L, so using ∇S as the Langevin drift is the
  cleanest correction. The DM only contributes a *good initial state* via the
  upsampled L_intermediate samples — that is the multi-grid bit.

MALA hyperparameter selection
-----------------------------
For a Gaussian target, optimal MALA acceptance is ~57% (Roberts et al. '98)
with step size h scaling as d^{-1/3} where d = number of degrees of freedom.
For phi^4 in normalized [-1,1] field space at L=128 (d=16384), this puts
h roughly in the range [5e-4, 5e-2]; we scan to find empirical optimum.

Run
---
  # 1) scan step sizes (recommended first)
  python multigrid_mala.py --run_dir <multi-L dir> --ep 6260 \
      --L_intermediate 64 --L_target 128 --scan_h --num_samples 256

  # 2) full run with the chosen h
  python multigrid_mala.py --run_dir <...> --ep 6260 \
      --L_intermediate 64 --L_target 128 --mala_h 1e-3 --mh_steps 500
"""

import sys
sys.path.append("..")

import argparse
import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


# ---------- phi^4 action / gradient on normalised fields -----------------------

def phi4_action(phi_norm, k, l, phi_min, phi_max):
    """phi_norm in [-1,1]; returns S per-config, shape (B,)."""
    p = (phi_norm[:, 0, :, :] + 1) / 2 * (phi_max - phi_min) + phi_min
    nb = torch.roll(p, 1, dims=1) + torch.roll(p, 1, dims=2)
    return torch.sum(-2 * k * p * nb + (1 - 2 * l) * p**2 + l * p**4,
                     dim=(1, 2))


def phi4_grad_S(phi_norm, k, l, phi_min, phi_max):
    """∂S/∂phi_norm (full neighbour sum), shape (B,1,L,L)."""
    scale = (phi_max - phi_min) / 2.0
    p = (phi_norm[:, 0, :, :] + 1) / 2 * (phi_max - phi_min) + phi_min
    nb = (torch.roll(p, 1, dims=1) + torch.roll(p, -1, dims=1)
          + torch.roll(p, 1, dims=2) + torch.roll(p, -1, dims=2))
    dS_dp = -2 * k * nb + 2 * (1 - 2 * l) * p + 4 * l * p**3
    return (dS_dp * scale).unsqueeze(1)


# ---------- one MALA step, drift = -∇S ----------------------------------------

def mala_step(x, h, action_fn, grad_S_fn):
    """Single MALA step. log π(x) = -S(x), so drift = -∇S.

    Returns (x_new, accept_bool) where x_new is x or proposal y.
    """
    grad_S_x = grad_S_fn(x)
    noise = torch.randn_like(x)
    y = x - h * grad_S_x + torch.sqrt(2 * h) * noise

    grad_S_y = grad_S_fn(y)
    # drift = h * (∇log π(x) + ∇log π(y)) = -h * (∇S(x) + ∇S(y))
    drift = -h * (grad_S_x + grad_S_y)
    log_q = (-(0.25 / h)
             * torch.sum((torch.sqrt(2 * h) * noise + drift) ** 2,
                         dim=(1, 2, 3))
             + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3)))
    log_pi = action_fn(x) - action_fn(y)  # = log π(y) - log π(x) (note sign)

    accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
    accept = torch.rand(x.shape[0], device=x.device) < accept_prob
    x_new = torch.where(accept[:, None, None, None], y, x)
    return x_new, accept.float()


# ---------- main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ep", type=str, required=True)
    p.add_argument("--k", type=float, default=0.2705)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--L_intermediate", type=int, default=64)
    p.add_argument("--L_target", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=2000,
                   help="SDE steps at L_intermediate")
    p.add_argument("--mh_steps", type=int, default=500)
    p.add_argument("--mala_h", type=float, default=None)
    p.add_argument("--scan_h", action="store_true")
    p.add_argument("--scan_steps", type=int, default=100,
                   help="MALA steps used during step-size scan")
    p.add_argument("--scan_grid", type=str,
                   default="1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2",
                   help="Comma-separated h values to scan")
    p.add_argument("--upsample_mode", type=str, default="bilinear",
                   choices=["bilinear", "nearest", "bicubic"])
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse_phase1", type=str, default=None,
                   help="If given, load phase-1 samples from this .npy "
                        "instead of running the SDE.")
    args = p.parse_args()

    assert args.L_target % args.L_intermediate == 0, \
        "L_target should be an integer multiple of L_intermediate"

    # Load checkpoint
    ckpt_paths = sorted(Path(f"{args.run_dir}/models").glob(f"*{args.ep}*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No ckpt matching '*{args.ep}*' in {args.run_dir}/models")
    ckpt_path = ckpt_paths[-1]
    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    sigma = float(h.get("sigma", 360.0))
    norm_min = float(h.get("norm_min"))
    norm_max = float(h.get("norm_max"))
    sd = ckpt.get("state_dict", {})
    l_cond_detected = any("L_embed" in kk for kk in sd.keys())
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]  "
          f"l_cond={l_cond_detected}")

    # Action functions (closed over physical norms)
    action_fn = functools.partial(phi4_action, k=args.k, l=args.l,
                                  phi_min=norm_min, phi_max=norm_max)
    grad_S_fn = functools.partial(phi4_grad_S, k=args.k, l=args.l,
                                  phi_min=norm_min, phi_max=norm_max)

    # ------ Phase 1: SDE at L_intermediate ----------------------------------
    if args.reuse_phase1 is not None:
        print(f"Loading phase-1 samples from {args.reuse_phase1}")
        arr = np.load(args.reuse_phase1)            # (L, L, N) physical
        # Convert physical -> normalised
        x_norm_np = 2 * (arr - norm_min) / (norm_max - norm_min) - 1
        # Crop / repeat to match num_samples
        N_avail = x_norm_np.shape[-1]
        if N_avail < args.num_samples:
            print(f"  WARNING: only {N_avail} reuse samples; using all")
            args.num_samples = N_avail
        else:
            x_norm_np = x_norm_np[..., :args.num_samples]
        samples_inter = (torch.from_numpy(x_norm_np)
                         .permute(2, 0, 1).float()
                         .unsqueeze(1).to(args.device))
        L_inter = samples_inter.shape[-1]
        assert L_inter == args.L_intermediate, \
            f"reuse samples have L={L_inter}, expected {args.L_intermediate}"
    else:
        # Build model and run SDE at L_intermediate
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        score_model = NCSNpp2D(marginal_prob_std_fn, l_cond=l_cond_detected)
        model = DiffusionModel.load_from_checkpoint(
            str(ckpt_path), score_model=score_model)
        model.L = args.L_intermediate
        model = model.to(args.device).eval()

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Phase 1: SDE at L={args.L_intermediate}, "
              f"num_samples={args.num_samples}, num_steps={args.num_steps}")
        samples_inter = model.sample(args.num_samples, args.num_steps,
                                     schedule="log")
        # Free the SDE model — we don't need it for Phase 2/3
        del model, score_model
        torch.cuda.empty_cache()

    print(f"  L={args.L_intermediate} samples shape: {tuple(samples_inter.shape)}  "
          f"range=[{samples_inter.min().item():.3f},{samples_inter.max().item():.3f}]")

    # ------ Phase 2: bilinear upsample to L_target -------------------------
    factor = args.L_target / args.L_intermediate
    samples_up = F.interpolate(samples_inter, scale_factor=factor,
                               mode=args.upsample_mode,
                               align_corners=False if args.upsample_mode != "nearest" else None)
    samples_up = samples_up.float()  # ensure fp32 for MALA precision
    print(f"Phase 2: upsampled to L={args.L_target} via {args.upsample_mode}, "
          f"shape={tuple(samples_up.shape)}")

    # Quick diagnostic on the upsampled state (before MALA)
    with torch.no_grad():
        S_up = action_fn(samples_up).cpu().numpy()
    print(f"  upsampled S/V: {S_up.mean()/(args.L_target**2):.4f} "
          f"± {S_up.std()/(args.L_target**2):.4f}")

    # ------ Phase 3a: optional step-size scan ------------------------------
    if args.scan_h:
        scan_grid = [float(s) for s in args.scan_grid.split(",")]
        print(f"\nPhase 3a: scanning h over {scan_grid}")
        scan_results = []
        for h_val in scan_grid:
            x = samples_up.clone()
            h_t = torch.tensor(h_val, device=args.device, dtype=x.dtype)
            total_accept = torch.zeros(x.shape[0], device=args.device)
            with torch.no_grad():
                for j in range(args.scan_steps):
                    x, acc = mala_step(x, h_t, action_fn, grad_S_fn)
                    total_accept += acc
            rate = (total_accept / args.scan_steps).cpu().numpy()
            S_after = action_fn(x).cpu().numpy()
            print(f"  h={h_val:.1e}  acc={rate.mean():.4f}  "
                  f"(min/max {rate.min():.3f}/{rate.max():.3f})  "
                  f"S/V={S_after.mean()/(args.L_target**2):.4f}")
            scan_results.append((h_val, rate.mean()))

        # pick h whose mean acceptance is closest to 0.55 (a touch above the
        # 0.574 Gaussian-target optimum to be conservative)
        target_acc = 0.55
        best_h = min(scan_results, key=lambda r: abs(r[1] - target_acc))[0]
        print(f"\nSelected h (closest to acc={target_acc}): {best_h:.1e}")
    else:
        best_h = args.mala_h
        if best_h is None:
            best_h = 1e-3
            print(f"\nNo --mala_h given; defaulting to {best_h:.1e}")

    # ------ Phase 3b: full MALA run ---------------------------------------
    print(f"\nPhase 3b: full MALA, h={best_h:.1e}, mh_steps={args.mh_steps}")
    x = samples_up.clone()
    h_t = torch.tensor(best_h, device=args.device, dtype=x.dtype)
    total_accept = torch.zeros(x.shape[0], device=args.device)
    log_interval = max(1, args.mh_steps // 10)

    pbar = tqdm(range(args.mh_steps), desc="MALA")
    with torch.no_grad():
        for j in pbar:
            x, acc = mala_step(x, h_t, action_fn, grad_S_fn)
            total_accept += acc
            if (j + 1) % log_interval == 0:
                running = (total_accept / (j + 1)).mean().item()
                pbar.set_postfix(acc=f"{running:.4f}")

    accept_rate = (total_accept / args.mh_steps).cpu().numpy()
    print(f"\nFinal MALA acceptance: mean={accept_rate.mean():.4f}  "
          f"min={accept_rate.min():.4f}  max={accept_rate.max():.4f}")

    # ------ Save ------------------------------------------------------------
    samples_phys = (x[:, 0].cpu().numpy() + 1) / 2 * (norm_max - norm_min) + norm_min
    out_dir = Path(args.run_dir) / "data_multigrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"multigrid_{args.L_intermediate}to{args.L_target}_"
           f"{args.upsample_mode}_h{best_h:.0e}_mh{args.mh_steps}_ep{args.ep}")
    npy = out_dir / f"samples_{tag}.npy"
    np.save(npy, samples_phys.transpose(1, 2, 0))
    print(f"Saved: {npy}  shape={samples_phys.transpose(1,2,0).shape}")
    np.savetxt(out_dir / f"accept_{tag}.dat", accept_rate)
    print(f"Saved acceptance rates: {out_dir / f'accept_{tag}.dat'}")


if __name__ == "__main__":
    main()
