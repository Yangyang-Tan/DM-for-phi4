"""
Ablation on L=16-only catastrophic cross-L failure: does it persist if we
(a) use ep=9111 instead of ep=10000, (b) disable EMA averaging?

Plus reference: multi-L 4-8-16 ep=10000 with EMA (the "good" baseline).

For each model+config: run reverse-SDE on L=64 with identical initial noise
and identical per-step noise, snapshot ⟨|M|⟩ and G(k_min) trajectories.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import functools
import contextlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std

L_SAMPLE = 64
N_CFG = 16
NUM_STEPS = 2000
SNAPSHOT_T = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sigma = ckpt["hyper_parameters"]["sigma"]
    score_model = NCSNpp3D(functools.partial(marginal_prob_std, sigma=sigma))
    model = DiffusionModel3D.load_from_checkpoint(ckpt_path, score_model=score_model)
    model.L = L_SAMPLE
    return model.to(device).eval(), float(sigma)


@torch.no_grad()
def trajectory(model, init_x, noise_seed, time_steps, dt_all, snapshots_at, use_ema, device):
    snapshots = {}
    x = init_x.clone()
    if 1.0 in snapshots_at:
        snapshots[1.0] = x.cpu().clone()
    gen = torch.Generator(device=device).manual_seed(noise_seed)

    ema_ctx = model.ema.average_parameters() if use_ema else contextlib.nullcontext()
    with ema_ctx, torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        batch_t = torch.empty(x.shape[0], device=device)
        for i in range(len(time_steps) - 1):
            t_now = float(time_steps[i].item())
            batch_t.fill_(t_now)
            g = model.diffusion_coeff_fn(time_steps[i])
            dt = dt_all[i]
            mean_x = x + g**2 * model(x, batch_t) * dt
            step_noise = torch.randn(x.shape, generator=gen, device=device, dtype=torch.float32)
            x = mean_x + g * torch.sqrt(dt) * step_noise
            t_next = float(time_steps[i + 1].item())
            for ts in list(snapshots_at):
                if t_now >= ts > t_next and round(ts, 6) not in snapshots:
                    snapshots[round(ts, 6)] = mean_x.cpu().clone()
    return snapshots


def stats(x):
    arr = x[:, 0].numpy().astype(np.float64)
    M = arr.mean(axis=(1, 2, 3))
    L = arr.shape[1]
    phi = arr - arr.mean()
    fk = np.fft.fftn(phi, axes=(1, 2, 3))
    g = (np.abs(fk) ** 2) / (L ** 3)
    diag_n = np.arange(1, L // 2 + 1)
    g_diag = g[:, diag_n, diag_n, diag_n].mean(axis=0)
    return {"abs_M": float(np.abs(M).mean()),
            "std":   float(arr.std()),
            "G_kmin": float(g_diag[0])}


def run_case(label, ckpt_path, use_ema, device, sigma_expected=None):
    print(f"\n=== {label} ===")
    model, sigma = load_model(ckpt_path, device)
    if sigma_expected is not None and abs(sigma - sigma_expected) > 1e-3:
        print(f"  WARNING sigma mismatch: ckpt={sigma}, expected={sigma_expected}")

    eps = 1e-5
    s = torch.linspace(0, 1, NUM_STEPS + 1, device=device)
    time_steps = (1.0 - eps) * (1 - s) + eps  # linear schedule
    dt_all = time_steps[:-1] - time_steps[1:]

    init_gen = torch.Generator(device=device).manual_seed(42)
    init_std = float(np.sqrt((sigma ** 2 - 1) / (2 * np.log(sigma))))
    init_x = torch.randn(N_CFG, 1, L_SAMPLE, L_SAMPLE, L_SAMPLE,
                         generator=init_gen, device=device) * init_std

    snaps = trajectory(model, init_x, 1234, time_steps, dt_all, SNAPSHOT_T, use_ema, device)

    rows = []
    print(f"{'t':>6}  {'⟨|M|⟩':>7}  {'std':>7}  {'G(k_min)':>10}")
    for ts in SNAPSHOT_T:
        if ts not in snaps:
            continue
        s_ = stats(snaps[ts])
        rows.append((ts, s_["abs_M"], s_["std"], s_["G_kmin"]))
        print(f"{ts:>6.3f}  {s_['abs_M']:>7.3f}  {s_['std']:>7.2f}  {s_['G_kmin']:>10.2f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--use_ema", type=int, choices=[0, 1], required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--out_pkl", type=str, required=True)
    args = p.parse_args()
    device = args.device

    rows = run_case(args.label, args.ckpt, bool(args.use_ema), device)
    import pickle
    with open(args.out_pkl, "wb") as f:
        pickle.dump({args.label: rows}, f)
    print(f"Saved data: {args.out_pkl}")
    return

    import pickle
    with open(args.out_pkl, "wb") as f:
        pickle.dump(all_rows, f)
    print(f"\nSaved data: {args.out_pkl}")
    return

    # Plot (kept for reference; main entrypoint exits above)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = {
        "L=16-only ep=10000 EMA":     ("C3", "-",  "o"),
        "L=16-only ep=10000 no-EMA":  ("C3", "--", "s"),
        "L=16-only ep=9111  EMA":     ("C1", "-",  "o"),
        "L=16-only ep=9111  no-EMA":  ("C1", "--", "s"),
        "multi-L 4-8-16 ep=10000 EMA":("C0", "-",  "D"),
        "multi-L 4-8-16 ep=9111  EMA":("C0", "--", "D"),
    }
    for label, rows in all_rows.items():
        ts   = [r[0] for r in rows]
        absM = [r[1] for r in rows]
        std_ = [r[2] for r in rows]
        Gkm  = [r[3] for r in rows]
        c, ls, mk = palette[label]
        axes[0].plot(ts, absM, marker=mk, ls=ls, color=c, lw=1.4, ms=5, label=label)
        axes[1].plot(ts, std_, marker=mk, ls=ls, color=c, lw=1.4, ms=5, label=label)
        axes[2].plot(ts, Gkm,  marker=mk, ls=ls, color=c, lw=1.4, ms=5, label=label)
    for ax, ylabel, title in zip(
        axes,
        [r"$\langle|M|\rangle$", r"std($\phi$)", r"$G(k_{\rm min})$"],
        ["⟨|M|⟩ along reverse-SDE trajectory",
         "std(φ) trajectory",
         "G(k_min) trajectory"]):
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("reverse-SDE time t"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        if "G" in title or "std" in title: ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"L=16-only failure ablation: ep=9111/10000 × EMA on/off  (κ=0.1923, L_sample=64, N={N_CFG})",
                 y=1.0, fontsize=11)
    plt.tight_layout()
    out = Path("results/sde_traj_L16_ablation.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
