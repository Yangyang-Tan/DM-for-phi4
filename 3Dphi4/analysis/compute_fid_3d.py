"""
Compute FID-like metrics for 3D phi4 field configurations.

Two methods:
  slice    — Extract 2D slices from 3D configs, compute standard FID via Inception v3.
  spectrum — Use radially-averaged 3D power spectrum as features, compute Fréchet distance.

Usage:
    python compute_fid_3d.py --epoch 12199
    python compute_fid_3d.py --epoch 12199 --method spectrum
    python compute_fid_3d.py --epoch 12199 --method slice --axis z --num_slices 8
    python compute_fid_3d.py --epoch 12199 --method all
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3


# ────────────────────────────── data loading ──────────────────────────────

def load_reference_data(path, max_samples=None):
    """Load HMC reference configs from JLD2/HDF5.  Returns (N, L, L, L)."""
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    if cfgs.ndim == 4 and cfgs.shape[-1] > cfgs.shape[0]:
        cfgs = cfgs.transpose(3, 0, 1, 2)
    if max_samples is not None:
        cfgs = cfgs[:max_samples]
    return cfgs


def load_generated_data(path, max_samples=None):
    """Load generated samples (.npy).  Returns (N, L, L, L)."""
    cfgs = np.load(path).astype(np.float32)
    if cfgs.ndim == 4 and cfgs.shape[-1] > cfgs.shape[0]:
        cfgs = cfgs.transpose(3, 0, 1, 2)
    if max_samples is not None:
        cfgs = cfgs[:max_samples]
    return cfgs


# ────────────────────────────── Fréchet distance ──────────────────────────

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# ─────────────────────────── Method 1: Slice FID ─────────────────────────

class InceptionFeatureExtractor:
    """Wrap Inception v3 to extract pool-3 (2048-d) features."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def extract(self, images, batch_size=64):
        """images: (N, H, W) float32 in physical range. Returns (N, 2048)."""
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch = torch.from_numpy(batch).float().to(self.device)
            batch = (batch - batch.min()) / (batch.max() - batch.min() + 1e-8)
            batch = batch.unsqueeze(1).expand(-1, 3, -1, -1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            out = self.model(batch)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


def extract_slices(cfgs, axis="z", num_slices=8, seed=42):
    """Extract 2D slices from 3D configs.  Returns (N*num_slices, L, L)."""
    rng = np.random.default_rng(seed)
    N, Lx, Ly, Lz = cfgs.shape
    ax = {"x": 1, "y": 2, "z": 3}[axis]
    L_ax = cfgs.shape[ax]
    indices = rng.choice(L_ax, size=min(num_slices, L_ax), replace=False)
    indices.sort()
    slices = np.take(cfgs, indices, axis=ax)
    slices = slices.reshape(-1, *[s for i, s in enumerate(cfgs.shape[1:]) if i != ax - 1])
    return slices


def compute_slice_fid(ref, gen, axis="z", num_slices=8, device="cuda", batch_size=64):
    """Compute FID on 2D slices extracted from 3D configs."""
    print(f"  Extracting {axis}-slices (num_slices={num_slices}) ...")
    ref_slices = extract_slices(ref, axis=axis, num_slices=num_slices)
    gen_slices = extract_slices(gen, axis=axis, num_slices=num_slices)
    print(f"  Reference slices: {ref_slices.shape}, Generated slices: {gen_slices.shape}")

    extractor = InceptionFeatureExtractor(device=device)
    print("  Computing Inception features for reference ...")
    feat_ref = extractor.extract(ref_slices, batch_size=batch_size)
    print("  Computing Inception features for generated ...")
    feat_gen = extractor.extract(gen_slices, batch_size=batch_size)

    mu1, sigma1 = feat_ref.mean(axis=0), np.cov(feat_ref, rowvar=False)
    mu2, sigma2 = feat_gen.mean(axis=0), np.cov(feat_gen, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


# ───────────────────────── Method 2: Spectrum FID ─────────────────────────

def radial_power_spectrum(cfg, n_bins=None):
    """Radially-averaged 3D power spectrum of a single (L,L,L) config."""
    L = cfg.shape[0]
    if n_bins is None:
        n_bins = L // 2
    ft = np.fft.fftn(cfg)
    ps = np.abs(ft) ** 2
    kx = np.fft.fftfreq(L, d=1.0) * L
    ky = np.fft.fftfreq(L, d=1.0) * L
    kz = np.fft.fftfreq(L, d=1.0) * L
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_bins = np.linspace(0, L // 2, n_bins + 1)
    spectrum = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (K >= k_bins[b]) & (K < k_bins[b + 1])
        if mask.any():
            spectrum[b] = ps[mask].mean()
    return spectrum


def config_features(cfg):
    """Feature vector for one 3D config: [log-power-spectrum, moments]."""
    ps = radial_power_spectrum(cfg)
    ps_feat = np.log1p(ps)
    m1 = cfg.mean()
    m2 = (cfg**2).mean()
    m4 = (cfg**4).mean()
    mag = np.abs(cfg).mean()
    return np.concatenate([ps_feat, [m1, m2, m4, mag]])


def compute_spectrum_fid(ref, gen):
    """Fréchet distance in physics-feature space (power spectrum + moments)."""
    print("  Computing power spectrum features for reference ...")
    feat_ref = np.array([config_features(ref[i]) for i in range(len(ref))])
    print("  Computing power spectrum features for generated ...")
    feat_gen = np.array([config_features(gen[i]) for i in range(len(gen))])
    print(f"  Feature dimension: {feat_ref.shape[1]}")

    mu1, sigma1 = feat_ref.mean(axis=0), np.cov(feat_ref, rowvar=False)
    mu2, sigma2 = feat_gen.mean(axis=0), np.cov(feat_gen, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


# ───────────────────────── Bonus: observable summary ──────────────────────

def print_observable_comparison(ref, gen):
    """Print side-by-side comparison of basic observables."""
    def stats(cfgs, label):
        mag = np.array([c.mean() for c in cfgs])
        sus = np.array([(c**2).mean() - c.mean()**2 for c in cfgs])
        abs_mag = np.array([np.abs(c).mean() for c in cfgs])
        print(f"  {label:12s}  <phi>={mag.mean():.6f}±{mag.std():.6f}  "
              f"<|phi|>={abs_mag.mean():.6f}±{abs_mag.std():.6f}  "
              f"chi={sus.mean():.4f}±{sus.std():.4f}")

    print("\n  Observable comparison:")
    stats(ref, "Reference")
    stats(gen, "Generated")


# ──────────────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FID for 3D phi4")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch tag, e.g. 12199")
    parser.add_argument("--method", type=str, default="all", choices=["slice", "spectrum", "all"])
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("--num_slices", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--k", type=float, default=0.2)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--max_ref", type=int, default=None, help="Limit reference samples")
    parser.add_argument("--max_gen", type=int, default=None, help="Limit generated samples")
    parser.add_argument("--ref_path", type=str, default=None, help="Override reference data path")
    parser.add_argument("--gen_path", type=str, default=None, help="Override generated data path")
    args = parser.parse_args()

    base = f"phi4_3d_L{args.L}_k{args.k}_l{args.l}_{args.network}"
    ref_path = args.ref_path or f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"
    gen_path = args.gen_path or f"{base}/data/samples_epoch={args.epoch}.npy"

    print(f"Reference : {ref_path}")
    print(f"Generated : {gen_path}")

    ref = load_reference_data(ref_path, max_samples=args.max_ref)
    gen = load_generated_data(gen_path, max_samples=args.max_gen)
    print(f"Reference shape: {ref.shape},  Generated shape: {gen.shape}")

    print_observable_comparison(ref, gen)

    if args.method in ("slice", "all"):
        print(f"\n[Slice FID] (axis={args.axis}, num_slices={args.num_slices})")
        fid_slice = compute_slice_fid(
            ref, gen, axis=args.axis, num_slices=args.num_slices,
            device=args.device, batch_size=args.batch_size,
        )
        print(f"  >>> Slice FID = {fid_slice:.4f}")

    if args.method in ("spectrum", "all"):
        print(f"\n[Spectrum FID] (3D power spectrum + moments)")
        fid_spec = compute_spectrum_fid(ref, gen)
        print(f"  >>> Spectrum FID = {fid_spec:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
