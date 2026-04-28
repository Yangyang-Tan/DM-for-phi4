"""
Test #2: post-hoc renormalisation with L=64 HMC norm range.

Hypothesis: the +5% bias in G(k) at κ=0.28 cross-L comes from using L=32
norm_min/max (baked into the saved samples). Reinterpret: invert L=32
denorm to get x_norm ∈ [-1,1], then re-denormalise with L=64 HMC range.

Mechanically this is just an affine rescaling
    φ_re = (φ - a32) / (b32 - a32) * (b64 - a64) + a64
so propagator scales by ((b64-a64)/(b32-a32))². The ⟨φ⟩ shift only affects
the zero-mode; G(k≠0) is unchanged by a constant shift.

Print the predicted/observed bias before and after.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py


def load_dm(path: Path) -> np.ndarray:
    return np.load(path).transpose(2, 0, 1).astype(np.float64)


def load_hmc(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs


def G_axial(cfgs: np.ndarray, n_x_list=(1, 2, 3, 5, 10)):
    L = cfgs.shape[1]
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V
    return [(4 * np.sin(np.pi * n / L) ** 2, pk[:, n, 0].mean()) for n in n_x_list]


cases = [("0.26", "4999", "symmetric"),
         ("0.2705", "9999", "near-critical"),
         ("0.28", "4999", "broken")]

# Cached norm ranges from ckpts
norm_train = {
    "0.26":   (-4.0807, 3.9009),
    "0.2705": (-4.3525, 4.1927),
    "0.28":   (-4.4513, 4.3591),
}

for k, ep, label in cases:
    print(f"\n=== {label}  k={k} ===")
    ref = load_hmc(Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_64^2.jld2"))[:8192]
    a64, b64 = float(ref.min()), float(ref.max())
    a32, b32 = norm_train[k]
    span_ratio = (b64 - a64) / (b32 - a32)
    print(f"   L=32 norm: [{a32:.4f},{b32:.4f}]  span={b32-a32:.4f}  std={ref.std():.4f}")
    print(f"   L=64 norm: [{a64:.4f},{b64:.4f}]  span={b64-a64:.4f}  span_ratio={span_ratio:.4f}")
    print(f"   Predicted G ratio after renorm: {span_ratio**2:.4f}")

    f_log = Path(f"phi4_L32_k{k}_l0.022_ncsnpp/data_crossL/"
                 f"samples_crossL_train32_sample64_em_log_steps2000_ep{ep}.npy")
    dm = load_dm(f_log)
    # Invert L=32 denorm, re-denorm with L=64
    x_norm = 2 * (dm - a32) / (b32 - a32) - 1
    dm_re = (x_norm + 1) / 2 * (b64 - a64) + a64

    print(f"   {'mode (n_x,0)':>14}  {'k̂²_x':>8}  {'HMC':>10}  {'DM[log]':>10}  ratio   {'DM[log] renorm':>14}  ratio")
    Gr = G_axial(ref); Gd = G_axial(dm); Gn = G_axial(dm_re)
    for (kk2, gr), (_, gd), (_, gn) in zip(Gr, Gd, Gn):
        print(f"   {'(_,0)':>14}  {kk2:8.4f}  {gr:10.4f}  {gd:10.4f}  {gd/gr:5.3f}    {gn:10.4f}  {gn/gr:5.3f}")
