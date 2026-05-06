"""
Dynamic experiment: snapshot reverse-SDE intermediate states for paired
(single-L, multi-L) models on L=64. Use IDENTICAL initial noise + IDENTICAL
per-step noise (seeded) so trajectories differ only via the score function.

Snapshots at fixed t. At each snapshot:
  - statistics (mean, std, ⟨|M|⟩) of x_t
  - diagonal G(|k̂|) of x_t  (still meaningful as Fourier moment)
  - check if/when the trajectories visibly diverge.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import functools
import numpy as np
import torch
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std

import argparse
L_SAMPLE = 64
N_CFG = 16              # small for memory + speed; same noise across pair
NUM_STEPS = 2000
SNAPSHOT_T = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005]
DEVICE = None  # set in main


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sigma = ckpt["hyper_parameters"]["sigma"]
    score_model = NCSNpp3D(functools.partial(marginal_prob_std, sigma=sigma))
    model = DiffusionModel3D.load_from_checkpoint(ckpt_path, score_model=score_model)
    model.L = L_SAMPLE
    return model.to(DEVICE).eval(), float(sigma)


@torch.no_grad()
def trajectory_with_snapshots(model, init_x, noise_seed, time_steps, dt_all, snapshot_t):
    """Reverse Euler-Maruyama. Per-step noise is generated on the fly via a
    seeded torch.Generator (same seed → same per-step noise across paired models).
    """
    snapshots = {}
    x = init_x.clone()
    if 1.0 in snapshot_t:
        snapshots[1.0] = x.cpu().clone()
    gen = torch.Generator(device=DEVICE).manual_seed(noise_seed)
    with model.ema.average_parameters(), torch.autocast(DEVICE.split(":")[0], dtype=torch.bfloat16):
        batch_t = torch.empty(x.shape[0], device=DEVICE)
        for i in range(len(time_steps) - 1):
            t_now = float(time_steps[i].item())
            batch_t.fill_(t_now)
            g = model.diffusion_coeff_fn(time_steps[i])
            dt = dt_all[i]
            mean_x = x + g**2 * model(x, batch_t) * dt
            step_noise = torch.randn(x.shape, generator=gen, device=DEVICE, dtype=torch.float32)
            x = mean_x + g * torch.sqrt(dt) * step_noise
            t_next = float(time_steps[i + 1].item())
            for ts in list(snapshot_t):
                if t_now >= ts > t_next and round(ts, 6) not in snapshots:
                    snapshots[round(ts, 6)] = mean_x.cpu().clone()
    return snapshots


def stats(x):
    """x: (N, 1, L, L, L) — return per-cfg stats averaged."""
    arr = x[:, 0].numpy().astype(np.float64)
    M = arr.mean(axis=(1, 2, 3))
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "abs_M":  float(np.abs(M).mean()),
        "M_std":  float(M.std()),
    }


def diagonal_propagator(arr):
    """Diagonal G(|k̂|) for arr shape (N, L, L, L)."""
    L = arr.shape[1]
    phi = arr - arr.mean()
    fk = np.fft.fftn(phi, axes=(1, 2, 3))
    g = (np.abs(fk) ** 2) / (L ** 3)
    diag_n = np.arange(1, L // 2 + 1)
    g_diag = g[:, diag_n, diag_n, diag_n]
    k_lat = np.sqrt(3.0) * 2.0 * np.sin(np.pi * diag_n / L)
    return k_lat, g_diag.mean(axis=0)


def run_pair(name_a, ckpt_a, name_b, ckpt_b, sigma_label, panel_title):
    print(f"\n=== Pair {name_a} ↔ {name_b} ===")
    m_a, sig_a = load_model(ckpt_a)
    m_b, sig_b = load_model(ckpt_b)
    assert abs(sig_a - sig_b) < 1e-3, f"sigma mismatch {sig_a} vs {sig_b}"
    sigma = sig_a

    # Build time schedule (linear)
    eps = 1e-5
    s = torch.linspace(0, 1, NUM_STEPS + 1, device=DEVICE)
    time_steps = (1.0 - eps) * (1 - s) + eps    # linear, decreasing
    dt_all = time_steps[:-1] - time_steps[1:]

    # Initial noise: same for both models
    init_gen = torch.Generator(device=DEVICE).manual_seed(42)
    init_std = float(np.sqrt((sigma**2 - 1) / (2*np.log(sigma))))
    init_x = torch.randn(N_CFG, 1, L_SAMPLE, L_SAMPLE, L_SAMPLE,
                         generator=init_gen, device=DEVICE) * init_std

    print(f"Running {name_a} (σ={sigma_label})...")
    snaps_a = trajectory_with_snapshots(m_a, init_x, 1234, time_steps, dt_all, SNAPSHOT_T)
    print(f"Running {name_b} (σ={sigma_label})...")
    snaps_b = trajectory_with_snapshots(m_b, init_x, 1234, time_steps, dt_all, SNAPSHOT_T)

    # Print stats and divergence at each snapshot
    print(f"\n{'t':>6}  {name_a:>15} {'⟨|M|⟩':>7} {'std':>7} {'G(k_min)':>10} | "
          f"{name_b:>15} {'⟨|M|⟩':>7} {'std':>7} {'G(k_min)':>10} | {'L2(Δx)':>9}")
    div = []
    for ts in SNAPSHOT_T:
        if ts not in snaps_a or ts not in snaps_b: continue
        xa, xb = snaps_a[ts], snaps_b[ts]
        sa, sb = stats(xa), stats(xb)
        ka, Ga = diagonal_propagator(xa[:, 0].numpy().astype(np.float64))
        _,  Gb = diagonal_propagator(xb[:, 0].numpy().astype(np.float64))
        l2 = float(((xa - xb) ** 2).mean().sqrt())
        div.append((ts, sa, sb, Ga[0], Gb[0], l2))
        print(f"{ts:>6.3f}  {'':>15} {sa['abs_M']:>7.3f} {sa['std']:>7.2f} {Ga[0]:>10.2f} | "
              f"{'':>15} {sb['abs_M']:>7.3f} {sb['std']:>7.2f} {Gb[0]:>10.2f} | {l2:>9.4f}")

    return {"name_a": name_a, "name_b": name_b, "sigma": sigma,
            "snaps_a": snaps_a, "snaps_b": snaps_b, "div": div}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--pair", type=int, default=0,
                   help="0 = both pairs serially, 1 = L=8 family only, 2 = L=16 family only")
    p.add_argument("--out_pkl", type=str, default=None,
                   help="If set, save div data to this pickle for later combined plotting")
    args = p.parse_args()
    global DEVICE
    DEVICE = args.device

    pair1 = pair2 = None
    if args.pair in (0, 1):
        pair1 = run_pair("L=8-only", "runs/phi4_3d_L8_k0.1923_l0.9_ncsnpp_sigma100/models/epoch=10000.ckpt",
                         "multi-L 4-8", "runs/phi4_3d_Lmulti4-8_k0.1923_l0.9_ncsnpp/models/epoch=10000.ckpt",
                         sigma_label=100, panel_title="L=8 family on L=64")

    if args.pair in (0, 2):
        pair2 = run_pair("L=16-only", "runs/phi4_3d_L16_k0.1923_l0.9_ncsnpp_sigma280/models/epoch=10000.ckpt",
                         "multi-L 4-8-16", "runs/phi4_3d_Lmulti4-8-16_k0.1923_l0.9_ncsnpp/models/epoch=10000.ckpt",
                         sigma_label=280, panel_title="L=16 family on L=64")

    if args.out_pkl:
        import pickle
        with open(args.out_pkl, "wb") as f:
            pickle.dump({"pair1": pair1, "pair2": pair2}, f)
        print(f"Saved data: {args.out_pkl}")
        return  # caller will combine and plot

    # Plot trajectory divergence
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, pair in enumerate([pair1, pair2]):
        ts = [d[0] for d in pair["div"]]
        absM_a = [d[1]["abs_M"] for d in pair["div"]]
        absM_b = [d[2]["abs_M"] for d in pair["div"]]
        std_a  = [d[1]["std"]   for d in pair["div"]]
        std_b  = [d[2]["std"]   for d in pair["div"]]
        Gkm_a  = [d[3]          for d in pair["div"]]
        Gkm_b  = [d[4]          for d in pair["div"]]
        l2     = [d[5]          for d in pair["div"]]

        # Panel A: ⟨|M|⟩
        ax = axes[row, 0]
        ax.plot(ts, absM_a, 'o-',  label=pair["name_a"],  lw=1.6, ms=6)
        ax.plot(ts, absM_b, 's--', label=pair["name_b"],  lw=1.6, ms=6)
        ax.set_xlabel("t (reverse-SDE time)")
        ax.set_ylabel(r"$\langle|M|\rangle$")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_title(f"⟨|M|⟩ trajectory (σ={pair['sigma']:.0f})")
        ax.legend(); ax.grid(alpha=0.3)

        # Panel B: std(φ)
        ax = axes[row, 1]
        ax.plot(ts, std_a, 'o-',  label=pair["name_a"],  lw=1.6, ms=6)
        ax.plot(ts, std_b, 's--', label=pair["name_b"],  lw=1.6, ms=6)
        ax.set_xlabel("t"); ax.set_ylabel(r"std($\phi$)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_title("std(φ) trajectory")
        ax.legend(); ax.grid(alpha=0.3, which="both")

        # Panel C: L2 distance between models' x_t
        ax = axes[row, 2]
        ax.plot(ts, l2, 'd-', color='C2', lw=1.8, ms=7,
                label=f"||x_a − x_b||₂ / sqrt(V·N)")
        ax.set_xlabel("t"); ax.set_ylabel("L2 per voxel")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_title("Trajectory divergence")
        ax.legend(); ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Reverse-SDE trajectory comparison on L={L_SAMPLE} (κ=0.1923, λ=0.9, "
        f"identical noise pair-wise, N={N_CFG} cfgs)",
        y=1.0, fontsize=12)
    plt.tight_layout()
    out = Path("results/sde_trajectory_compare.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
