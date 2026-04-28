"""
Quick one-off: em sampling at ep=10000 with schedule='linear' (vs default 'log'),
N=128 samples, for the 3D sigma=2760 model. Checks whether schedule change
improves the propagator UV mismatch we observed.

Output: samples_em_linear_steps2000_epoch=10000.npy  (same format as sweep)
"""
import sys, os, functools, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   # 3Dphi4/
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # repo root

import numpy as np
import torch

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std


RUN_DIR = "/data/tyywork/DM/3Dphi4/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760"
CKPT    = f"{RUN_DIR}/models/epoch=10000.ckpt"
DEVICE  = "cuda:2"
SEED    = 20260422
NUM_SAMPLES = 128
SDE_STEPS   = 2000
SCHEDULE    = "linear"

torch.set_float32_matmul_precision("medium")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
hp = ck.get("hyper_parameters", {})
sigma = float(hp["sigma"]); norm_min = float(hp["norm_min"]); norm_max = float(hp["norm_max"])
print(f"[lin] ckpt sigma={sigma} L={hp['L']} norm=[{norm_min:.4f}, {norm_max:.4f}]")
del ck

mps = functools.partial(marginal_prob_std, sigma=sigma)
raw = NCSNpp3D(mps)
compiled = torch.compile(raw, mode="reduce-overhead")
print(f"[lin] compiled NCSNpp3D")

dm = DiffusionModel3D.load_from_checkpoint(CKPT, score_model=compiled)
dm = dm.to(DEVICE).eval()

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
t0 = time.time()
samples = dm.sample(NUM_SAMPLES, SDE_STEPS, schedule=SCHEDULE)
dt = time.time() - t0
print(f"[lin] sampled {NUM_SAMPLES} with schedule={SCHEDULE} in {dt:.1f}s "
      f"({SDE_STEPS/dt:.2f} it/s)")

# Renormalize to training data range (same as sweep)
x = samples[:, 0].cpu().numpy()
x = (x + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
x = x.transpose(1, 2, 3, 0)  # (L, L, L, N)

out = f"{RUN_DIR}/data/samples_em_linear_steps{SDE_STEPS}_epoch=10000.npy"
np.save(out, x)
print(f"[lin] saved {out}  shape={x.shape}  range=[{x.min():.3f},{x.max():.3f}]")
