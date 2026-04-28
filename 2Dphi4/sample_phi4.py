"""
Generate samples from a trained 2D phi4 diffusion model.

Usage:
    python sample_phi4.py
    python sample_phi4.py --num_samples 64 --method mala
"""

import sys
sys.path.append("..")

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

from networks import ScoreNet, ScoreNetUNetPeriodic, NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from phi4_action import phi4_action as _phi4_action_unified, phi4_grad_S as _phi4_grad_S_unified


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


def _prepare_site_data(data, dtype=np.float64, lattice_axes=(-2, -1)):
    """Return (x_flat, site_shape) with x_flat shaped (n_samples, n_sites)."""
    x = np.asarray(data)
    if x.ndim == 4 and x.shape[1] == 1:
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
    
    # Per-bin, per-site moments
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


def phi4_action(phi, k, l, phi_min, phi_max):
    """2D phi^4 action (delegates to shared ``phi4_action`` module)."""
    return _phi4_action_unified(phi, k, l, phi_min, phi_max, spatial_dims=2)


def phi4_grad_S(phi, k, l, phi_min, phi_max):
    """∂S/∂x_norm for 2D phi^4 (delegates to shared module)."""
    return _phi4_grad_S_unified(phi, k, l, phi_min, phi_max, spatial_dims=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--method", type=str, default="em")
    parser.add_argument("--ep", type=str, default=None)
    # parser.add_argument("--output", type=str, default="phi4_samples_1")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--plot_grid", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--t_mh", type=float, default=0.01)
    parser.add_argument("--mh_steps", type=int, default=200)
    parser.add_argument("--rescale", action="store_true", help="Rescale samples by variance ratio")
    parser.add_argument("--data_path", type=str, default=None, help="Training data path for rescaling")
    parser.add_argument("--network", type=str, default="ncsnpp",
                        choices=["scorenet", "unet", "ncsnpp"],
                        help="Network architecture: scorenet | unet | ncsnpp")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix on the training output dir (e.g. '_sigma150')")
    parser.add_argument("--ode_method", type=str, default="dpm2",
                        choices=["dpm1", "dpm2", "dpm3", "rk45"],
                        help="ODE solver when --method=ode (default dpm2)")
    parser.add_argument("--n_repeats", type=int, default=4,
                        help="Number of independent sampling passes to concatenate")
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, seeds torch/cuda RNG before each sampling pass. "
                             "Fixes both the initial noise x_T and all per-step noise "
                             "in SDE reverse integration, giving identical trajectories "
                             "across calls. Repeats use seed, seed+1, ... (deterministic).")
    args = parser.parse_args()

    run_dir = f"phi4_L{args.L}_k{args.k}_l{args.l}_{args.network}{args.output_suffix}"

    # Get checkpoint
    if args.checkpoint is None:
        ckpts = sorted(Path(f"{run_dir}/models").glob(f"*{args.ep}*.ckpt"))
        args.checkpoint = str(ckpts[-1]) if ckpts else None
    print(f"Checkpoint: {args.checkpoint}")

    # Load hparams from checkpoint first
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    L = hparams.get("L", args.L)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"norm_min: {norm_min}, norm_max: {norm_max}")
    output = f"{run_dir}/data/"
    if not os.path.exists(output):
        os.makedirs(output)
    output = os.path.join(output, "samples")
    # Parse k, l from checkpoint path (directory or filename)
    # e.g., phi4_L128_k0.5_l0.022/models/00-105.ckpt or phi4_L128_k0.5_l0.022-epoch.ckpt
    ckpt_path = str(args.checkpoint)
    k_match = re.search(r'_k([\d.]+)', ckpt_path)
    l_match = re.search(r'_l([\d.]+)', ckpt_path)
    k = float(k_match.group(1)) if k_match else args.k
    l = float(l_match.group(1)) if l_match else args.l
    print(f"L={L}, k={k}, l={l}, sigma={sigma}")

    # Load model with correct sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if args.network == "ncsnpp":
        score_model = NCSNpp2D(marginal_prob_std_fn)
    elif args.network == "scorenet":
        score_model = ScoreNet(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNetUNetPeriodic(marginal_prob_std_fn)
    model = DiffusionModel.load_from_checkpoint(args.checkpoint, score_model=score_model)
    model = model.to(args.device).eval()

    # Sample
    print(f"Sampling ({args.method.upper()})  num_steps={args.num_steps}  n_repeats={args.n_repeats}  num_samples/rep={args.num_samples}")
    if args.seed is not None:
        print(f"Seed: {args.seed} (fixed IC + reverse-step noise)")

    def _seed_for(rep_idx):
        if args.seed is not None:
            s = args.seed + rep_idx
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)

    if args.method == "em":
        reps = []
        for i in range(args.n_repeats):
            _seed_for(i)
            reps.append(model.sample(args.num_samples, args.num_steps, schedule='log'))
        samples = torch.concatenate(reps, axis=0)
    elif args.method == "ode":
        reps = []
        for i in range(args.n_repeats):
            _seed_for(i)
            reps.append(model.sample_ode(args.num_samples, args.num_steps,
                                         schedule='log', method=args.ode_method))
        samples = torch.concatenate(reps, axis=0)
    elif args.method == "em2":
        samples = model.sample2(args.num_samples, args.num_steps)
        np.save("data/phi4_L128_k0.5_l0.022_t=0.2.npy", samples.detach().cpu().numpy())
        breakpoint()
    elif args.method == "pc":
        samples1 = model.sample_pc(args.num_samples, args.num_steps)
        samples2 = model.sample_pc(args.num_samples, args.num_steps)
        samples=torch.concatenate([samples1, samples2],axis=0)
        # samples = torch.concatenate([samples1, samples2],axis=0)
    else:  # mala
        action_fn = functools.partial(phi4_action, k=k, l=l, phi_min=norm_min, phi_max=norm_max)
        samples, acc = model.sample_mala(
            args.num_samples, args.num_steps, args.t_mh, args.mh_steps,
            action_fn=action_fn, schedule='log',
        )
        print(f"Acceptance rate (per-sample mean): {acc.mean().item():.4f}")
        print(f"Acceptance rate (min/max): {acc.min().item():.4f} / {acc.max().item():.4f}")
        np.savetxt(f"data/phi4_L{args.L}_k{args.k}_l{args.l}_t=10_mala_accept.dat", acc.detach().cpu().numpy())
        breakpoint()
    # Keep samples in normalized [-1, 1] range first
    # Shape: (num_samples, L, L)
    samples_norm = samples[:, 0].cpu().numpy()
    
    # Add Z2 symmetric samples
    # samples_with_z2 = np.concatenate([samples_norm, -samples_norm], axis=0)
    samples_with_z2 = samples_norm
    
    # Rescale by variance ratio if requested (on normalized data)
    # if args.rescale:
    if False:
        # Load training data and normalize
        print(f"Loading training data for rescaling: {args.data_path}")
        with h5py.File(args.data_path, "r") as f:
            cfgs_train = np.array(f["cfgs"])
        # Normalize training data to [-1, 1]
        cfgs_train_norm = ((cfgs_train - norm_min) / (norm_max - norm_min) - 0.5) * 2
        
        # Compute variance (2nd cumulant) using bootstrap
        cumulants_train, _ = lattice_bootstrap_cumulants(cfgs_train_norm, order=2, n_boot=10)
        cumulants_dm, _ = lattice_bootstrap_cumulants(samples_with_z2, order=2, n_boot=10)
        var_train = cumulants_train[1]  # κ_2 = variance
        var_dm = cumulants_dm[1]
        rescale_factor = np.sqrt(var_train / var_dm)
        
        print(f"Variance (κ_2, normalized) - Train: {var_train:.4f}, DM: {var_dm:.4f}")
        print(f"Rescale factor: {rescale_factor:.4f}")
        
        samples_with_z2 = samples_with_z2 * rescale_factor
    
    # Renormalize to original range at the end
    samples_renorm = (samples_with_z2 + 1) / 2 * (norm_max - norm_min) + norm_min
    
    # Save as (L, L, num_samples*n_repeats)
    samples_out = samples_renorm.transpose(1, 2, 0)
    tag = f"{args.method}_steps{args.num_steps}_{args.ep}"
    np.save(f"{output}_{tag}.npy", samples_out)
    print(f"Saved samples to {output}_{tag}.npy, shape: {samples_out.shape}")

    # Plot grid (only first plot_grid^2 samples)
    n = args.plot_grid
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, 0].cpu().numpy(), cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output}_{tag}.png", dpi=150)
    print(f"Saved plot to {output}_{tag}.png")


if __name__ == "__main__":
    main()
