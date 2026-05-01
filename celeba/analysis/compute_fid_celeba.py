"""
Compute FID for CelebA diffusion samples vs the preprocessed real dataset.

Uses torchvision Inception v3 pool-3 (2048-d) features + Frechet distance.
Grayscale images are replicated to 3 channels, bilinearly resized to 299x299,
and scaled to [0, 1] before inception's `transform_input=True` handling.

Data conventions in this repo
-----------------------------
Real: data/celeba_gray{64,128}.npy   shape (N, 1, H, W), float32 in [0, 1]
Gen : celeba_{L}_ncsnpp/data/samples_{method}_epoch=<ep>.npy
      shape (H, W, N), float32 roughly in [-1, 1]

Examples
--------
    # Single epoch/method
    python compute_fid_celeba.py --size 64 --method em --ep 9999 \
        --n_real 10000 --n_gen 0 --device cuda:2

    # Sweep every samples_<method>_epoch=*.npy in the model's data dir,
    # write results to fid_<method>.csv
    python compute_fid_celeba.py --size 128 --method em --sweep --device cuda:2
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3


# ---------------------------- data loading ----------------------------

def load_real(size, n=None, data_dir="data", seed=0):
    """Real: (N, 1, H, W) in [0, 1]  ->  (N, H, W) in [0, 1]."""
    path = os.path.join(data_dir, f"celeba_gray{size}.npy")
    arr = np.load(path, mmap_mode="r")
    if n is not None and n > 0 and n < len(arr):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(arr), size=n, replace=False)
        idx.sort()
        arr = np.asarray(arr[idx], dtype=np.float32)
    else:
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr  # (N, H, W) in [0, 1]


def load_gen(path, n=None):
    """Gen: (H, W, N) in [-1, 1]  ->  (N, H, W) in [0, 1] (clipped)."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == arr.shape[1] and arr.shape[-1] != arr.shape[0]:
        # (H, W, N) -> (N, H, W)
        arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if n is not None and n > 0 and n < len(arr):
        arr = arr[:n]
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return arr  # (N, H, W) in [0, 1]


# ---------------------------- Frechet distance ------------------------

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between N(mu1, sigma1) and N(mu2, sigma2).

    Uses the identity  tr(√(Σ1 Σ2)) = Σᵢ √λᵢ  where λᵢ are eigenvalues of
    the symmetric matrix √Σ1 · Σ2 · √Σ1. This is O(d³) once, robust to
    rank-deficient Σ (which happens when N ≤ feature_dim), and avoids the
    iterative `scipy.linalg.sqrtm` that can silently spin on CPU for
    an hour when the covariance-product is ill-conditioned.
    """
    diff = mu1 - mu2
    offset = eps * np.eye(sigma1.shape[0])
    s1 = sigma1 + offset
    s2 = sigma2 + offset
    # √Σ1 via eigendecomposition (Σ1 is symmetric PSD)
    w1, V1 = np.linalg.eigh(s1)
    w1 = np.clip(w1, 0.0, None)
    s1_sqrt = (V1 * np.sqrt(w1)) @ V1.T
    # Form symmetric matrix whose eigenvalues are eig(Σ1 Σ2)
    M = s1_sqrt @ s2 @ s1_sqrt
    M = 0.5 * (M + M.T)                      # symmetrize against FP asymmetry
    w = np.linalg.eigvalsh(M)
    w = np.clip(w, 0.0, None)
    tr_sqrt = np.sum(np.sqrt(w))
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_sqrt)


# ---------------------------- Inception feature extractor -------------

class InceptionFeats:
    """Inception v3 pool-3 (2048-d) features. Input: (N, H, W) in [0, 1]."""

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        # transform_input=True: torchvision normalizes [0,1] input internally.
        model = inception_v3(weights="IMAGENET1K_V1", transform_input=True, aux_logits=True)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, images, batch_size=64):
        feats = []
        for i in range(0, len(images), batch_size):
            b = torch.from_numpy(images[i : i + batch_size]).float().to(self.device)
            b = b.unsqueeze(1).expand(-1, 3, -1, -1)  # gray -> 3-ch
            b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
            out = self.model(b)
            feats.append(out.detach().float().cpu().numpy())
        return np.concatenate(feats, axis=0)


def stats(feat):
    return feat.mean(axis=0), np.cov(feat, rowvar=False)


# ---------------------------- main routines ---------------------------

def fid_for_path(gen_path, real_stats, extractor, n_gen=None, batch_size=64):
    gen = load_gen(gen_path, n=n_gen)
    feat = extractor(gen, batch_size=batch_size)
    mu_g, sigma_g = stats(feat)
    mu_r, sigma_r = real_stats
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g), len(gen)


EPOCH_RE = re.compile(r"epoch=(?:epoch=)?(\d+)")


def parse_ep(path):
    m = EPOCH_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, choices=[64, 128])
    p.add_argument("--network", type=str, default="ncsnpp")
    p.add_argument("--method", type=str, default="em",
                   help="em, dpm2, ... (matches samples_<method>_epoch=*.npy)")
    p.add_argument("--ep", type=str, default=None,
                   help="Single epoch tag, e.g. 9999 or epoch=9999. Ignored if --sweep.")
    p.add_argument("--sweep", action="store_true",
                   help="Compute FID for every samples_<method>_epoch=*.npy found.")
    p.add_argument("--gen_path", type=str, default=None,
                   help="Explicit path, overrides --ep/--sweep/--method.")
    p.add_argument("--n_real", type=int, default=10000,
                   help="Number of real images (0 = all ~200k).")
    p.add_argument("--n_gen", type=int, default=0,
                   help="Cap on generated samples per file (0 = all available).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", type=str, default=None,
                   help="CSV output (default: celeba_{L}_{net}/fid_{method}.csv in sweep mode).")
    args = p.parse_args()

    model_dir = f"celeba_{args.size}_{args.network}"

    # 1) Build real-data statistics once.
    n_real = None if args.n_real == 0 else args.n_real
    t0 = time.time()
    print(f"[real] loading {args.n_real or 'all'} from {args.data_dir}/celeba_gray{args.size}.npy")
    real = load_real(args.size, n=n_real, data_dir=args.data_dir, seed=args.seed)
    print(f"[real] shape={real.shape}  range=[{real.min():.3f},{real.max():.3f}]")

    extractor = InceptionFeats(device=args.device)
    print(f"[real] extracting Inception features ...")
    feat_r = extractor(real, batch_size=args.batch_size)
    mu_r, sigma_r = stats(feat_r)
    print(f"[real] done in {time.time()-t0:.1f}s, feat dim={feat_r.shape[1]}")
    del real, feat_r

    # 2) Decide which gen files to process.
    n_gen = None if args.n_gen == 0 else args.n_gen
    if args.gen_path:
        paths = [args.gen_path]
    elif args.sweep:
        pat = os.path.join(model_dir, "data", f"samples_{args.method}_epoch=*.npy")
        paths = sorted(glob.glob(pat), key=parse_ep)
        if not paths:
            sys.exit(f"No files match: {pat}")
        print(f"[sweep] {len(paths)} files matching {pat}")
    else:
        if args.ep is None:
            sys.exit("Need --ep, --sweep, or --gen_path.")
        ep = args.ep.replace("epoch=", "")
        pat = os.path.join(model_dir, "data", f"samples_{args.method}_epoch={ep}.npy")
        found = sorted(glob.glob(pat))
        if not found:
            sys.exit(f"No file matches: {pat}")
        paths = found

    # 3) Per-file FID.
    results = []
    for path in paths:
        ep = parse_ep(path)
        t = time.time()
        try:
            fid, n = fid_for_path(path, (mu_r, sigma_r), extractor,
                                  n_gen=n_gen, batch_size=args.batch_size)
        except Exception as e:
            print(f"  ep={ep:>6}  {os.path.basename(path)}  FAILED: {e}")
            continue
        dt = time.time() - t
        results.append((ep, path, n, fid))
        print(f"  ep={ep:>6}  n_gen={n:>5}  FID={fid:10.4f}  ({dt:.1f}s)  {os.path.basename(path)}")

    # 4) Save CSV.
    if results and (args.sweep or args.out_csv):
        out_csv = args.out_csv or os.path.join(model_dir, f"fid_{args.method}.csv")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w") as f:
            f.write("epoch,n_gen,fid,path\n")
            for ep, path, n, fid in sorted(results):
                f.write(f"{ep},{n},{fid:.6f},{path}\n")
        print(f"\n[csv] wrote {out_csv}")


if __name__ == "__main__":
    main()
