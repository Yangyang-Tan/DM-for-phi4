"""
Generate samples from a trained 3D phi4 diffusion model.

Usage:
    python sample_phi4.py
    python sample_phi4.py --num_samples 64 --method mala
    python sample_phi4.py --checkpoint path/to/ckpt --ep 0500
"""

import os
import re
import functools
import argparse
from pathlib import Path

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

from networks_3d import ScoreNet3D, ScoreNet3DUNetPeriodic, NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std


# ===== Bootstrap cumulants functions =====
def cumulants_from_moments(moments):
    """Cumulants from moments using the recursion."""
    m = np.asarray(moments, dtype=np.float64)
    order = m.shape[0]
    kappa = np.empty_like(m, dtype=np.float64)
    kappa[0] = m[0]
    for n in range(2, order + 1):
        val = m[n - 1].copy()
        for i in range(1, n):
            val -= binom(n - 1, i - 1) * kappa[i - 1] * m[n - i - 1]
        kappa[n - 1] = val
    return kappa


def _prepare_site_data(data, dtype=np.float64, lattice_axes=(-3, -2, -1)):
    """Return (x_flat, site_shape) with x_flat shaped (n_samples, n_sites)."""
    x = np.asarray(data)
    if x.ndim == 5 and x.shape[1] == 1:
        x = x[:, 0]
    x = np.asarray(x, dtype=dtype)
    nd = x.ndim
    lat_axes = tuple(ax if ax >= 0 else nd + ax for ax in lattice_axes)
    other_axes = tuple(ax for ax in range(1, nd) if ax not in lat_axes)
    if other_axes:
        x = x.mean(axis=other_axes, dtype=np.float64)
    site_shape = x.shape[1:]
    x_flat = x.reshape(x.shape[0], -1)
    return x_flat, site_shape


def lattice_bootstrap_cumulants(data, order=2, n_boot=100, seed=None, n_bins=100):
    """Bootstrap estimate of local cumulants κ_n. Returns (means, errors)."""
    rng = np.random.default_rng(seed)
    x_flat, _ = _prepare_site_data(data)
    n_samples, n_sites = x_flat.shape

    n_bins = int(min(n_bins, n_samples))
    bin_size = n_samples // n_bins
    n_used = bin_size * n_bins
    if n_used != n_samples:
        x_flat = x_flat[:n_used]

    per_bin_m = np.empty((n_bins, order, n_sites), dtype=np.float64)
    for b in range(n_bins):
        xb = x_flat[b * bin_size : (b + 1) * bin_size].astype(np.float64)
        x_pow = xb.copy()
        for k in range(order):
            per_bin_m[b, k] = x_pow.mean(axis=0)
            if k != order - 1:
                x_pow *= xb

    boot_kappa = np.empty((n_boot, order), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_bins, size=n_bins)
        m = per_bin_m[idx].mean(axis=0)
        kappa = cumulants_from_moments(m)
        boot_kappa[b] = kappa.mean(axis=1)

    return boot_kappa.mean(axis=0), boot_kappa.std(axis=0)


def phi4_action_3d(phi, k, l, phi_min, phi_max):
    """Compute 3D phi4 action (with denormalization)."""
    p = (phi[:, 0, :, :, :] + 1) / 2 * (phi_max - phi_min) + phi_min
    neighbor_sum = (torch.roll(p, 1, dims=1) +
                    torch.roll(p, 1, dims=2) +
                    torch.roll(p, 1, dims=3))
    return torch.sum(-2 * k * p * neighbor_sum + (1 - 2 * l) * p**2 + l * p**4, dim=(1, 2, 3))


def phi4_grad_S_3d(phi, k, l, phi_min, phi_max):
    """∂S/∂x_norm for 3D phi4, returned in normalised-field space (N,1,L,L,L)."""
    scale = (phi_max - phi_min) / 2.0
    p = (phi[:, 0, :, :, :] + 1) / 2 * (phi_max - phi_min) + phi_min
    nb = (torch.roll(p, 1, dims=1) + torch.roll(p, -1, dims=1)
        + torch.roll(p, 1, dims=2) + torch.roll(p, -1, dims=2)
        + torch.roll(p, 1, dims=3) + torch.roll(p, -1, dims=3))
    dS_dp = -2 * k * nb + 2 * (1 - 2 * l) * p + 4 * l * p ** 3
    return (dS_dp * scale).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--method", type=str, default="em")
    parser.add_argument("--ep", type=str, default=None)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--k", type=float, default=0.2)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--plot_grid", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t_mh", type=float, default=0.01)
    parser.add_argument("--mh_steps", type=int, default=200)
    parser.add_argument("--rescale", action="store_true", help="Rescale samples by variance ratio")
    parser.add_argument("--data_path", type=str, default=None, help="Training data path for rescaling")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["simple", "unet", "ncsnpp"],
                        help="Network architecture: simple | unet | ncsnpp")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix on the training output dir (e.g. '_sigma850')")
    parser.add_argument("--ode_method", type=str, default="dpm2",
                        choices=["dpm1", "dpm2", "dpm3"],
                        help="ODE solver when --method=ode")
    parser.add_argument("--schedule", type=str, default="log",
                        help="Time schedule for sampling")
    parser.add_argument("--n_repeats", type=int, default=4,
                        help="Number of independent sampling passes to concatenate")
    args = parser.parse_args()

    run_dir = f"phi4_3d_L{args.L}_k{args.k}_l{args.l}_{args.network}{args.output_suffix}"

    # Get checkpoint
    if args.checkpoint is None:
        ckpts = sorted(Path(f"{run_dir}/models").glob(f"*{args.ep}*.ckpt"))
        args.checkpoint = str(ckpts[-1]) if ckpts else None
    print(f"Checkpoint: {args.checkpoint}")

    # Load hparams from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    L = hparams.get("L", args.L)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"norm_min: {norm_min}, norm_max: {norm_max}")

    # Output directory
    output = f"{run_dir}/data/"
    if not os.path.exists(output):
        os.makedirs(output)
    output = os.path.join(output, "samples")

    # Parse k, l from checkpoint path
    ckpt_path = str(args.checkpoint)
    k_match = re.search(r'_k([\d.]+)', ckpt_path)
    l_match = re.search(r'_l([\d.]+)', ckpt_path)
    k = float(k_match.group(1)) if k_match else args.k
    l = float(l_match.group(1)) if l_match else args.l
    print(f"L={L}, k={k}, l={l}, sigma={sigma}")

    # Load model
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "simple":
        score_model = ScoreNet3D(marginal_prob_std_fn)
    elif args.network == "unet":
        score_model = ScoreNet3DUNetPeriodic(marginal_prob_std_fn)
    elif args.network == "ncsnpp":
        score_model = NCSNpp3D(marginal_prob_std_fn)
    print(f"Using network: {args.network}")
    model = DiffusionModel3D.load_from_checkpoint(args.checkpoint, score_model=score_model)
    model = model.to(args.device).eval()

    # Sample
    print(f"Sampling ({args.method.upper()})  steps={args.num_steps}  n_repeats={args.n_repeats}  samples/rep={args.num_samples}")
    if args.method == "em":
        reps = [model.sample(args.num_samples, args.num_steps, schedule=args.schedule)
                for _ in range(args.n_repeats)]
        samples = torch.concatenate(reps, axis=0)
    elif args.method == "ode":
        reps = [model.sample_ode(args.num_samples, args.num_steps,
                                 schedule=args.schedule, method=args.ode_method)
                for _ in range(args.n_repeats)]
        samples = torch.concatenate(reps, axis=0)
    elif args.method == "pc":
        reps = [model.sample_pc(args.num_samples, args.num_steps, schedule=args.schedule)
                for _ in range(args.n_repeats)]
        samples = torch.concatenate(reps, axis=0)
    else:  # mala
        action_fn = functools.partial(phi4_action_3d, k=k, l=l, phi_min=norm_min, phi_max=norm_max)
        samples, acc = model.sample_mala(
            args.num_samples, args.num_steps, args.t_mh, args.mh_steps,
            action_fn=action_fn, schedule='log',
        )
        print(f"Acceptance rate (per-sample mean): {acc.mean().item():.4f}")
        print(f"Acceptance rate (min/max): {acc.min().item():.4f} / {acc.max().item():.4f}")
        np.savetxt(f"data/phi4_3d_L{args.L}_k{args.k}_l{args.l}_t=10_mala_accept.dat", acc.detach().cpu().numpy())

    # Shape: (num_samples, L, L, L)
    samples_norm = samples[:, 0].cpu().numpy()
    samples_with_z2 = samples_norm

    # Renormalize to original range
    samples_renorm = (samples_with_z2 + 1) / 2 * (norm_max - norm_min) + norm_min

    # Save as (L, L, L, num_samples)
    samples_out = samples_renorm.transpose(1, 2, 3, 0)
    tag = f"{args.method}_steps{args.num_steps}_{args.ep}"
    np.save(f"{output}_{tag}.npy", samples_out)
    print(f"Saved samples to {output}_{tag}.npy, shape: {samples_out.shape}")

    # Plot 2D slices (middle z-slice)
    n = args.plot_grid
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    mid_z = L // 2
    for i, ax in enumerate(axes.flatten()):
        if i < samples.shape[0]:
            ax.imshow(samples[i, 0, mid_z].cpu().numpy(), cmap="viridis")
        ax.axis("off")
    plt.suptitle(f"3D phi4 samples (z={mid_z} slice)")
    plt.tight_layout()
    plt.savefig(f"{output}_{tag}.png", dpi=150)
    print(f"Saved plot to {output}_{tag}.png")


if __name__ == "__main__":
    main()
