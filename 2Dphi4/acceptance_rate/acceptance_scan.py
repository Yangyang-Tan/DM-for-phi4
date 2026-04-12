"""
Compute average MALA acceptance rate for all (L, k) combinations.

Average is taken over checkpoints in epoch range [epoch_lo, epoch_hi].
Fixed parameters: c (h=c/L²), t_mh=1e-4.

Usage:
    python acceptance_scan.py --device cuda:0
    python acceptance_scan.py --device cuda:0 --epoch_lo 3999 --epoch_hi 4999
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import re
import functools
import argparse
from pathlib import Path

import io

import h5py
import torch
import numpy as np

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from sample_phi4 import phi4_action


def find_epoch_checkpoints(model_dir):
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=epoch=*.ckpt")):
        if '-v' in f.stem:
            continue
        m = re.search(r'epoch=(\d+)', f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19

    path_str = str(ckpt_path)
    k = float(re.search(r'_k([\d.]+)', path_str).group(1))
    l = float(re.search(r'_l([\d.]+)', path_str).group(1))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel.load_from_checkpoint(buf, score_model=score_model)
    model = model.to(device).eval()
    action_fn = functools.partial(
        phi4_action, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )
    return model, action_fn, norm_min, norm_max


def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def measure_acceptance(model, action_fn, x_ref, h, t_mh, mh_steps):
    device = model.device
    num_samples = x_ref.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x_ref.dtype)
    x = x_ref.clone()
    batch_t = torch.ones(num_samples, device=device) * t_mh
    total_accept = torch.zeros(num_samples, device=device)

    with torch.no_grad():
        for _ in range(mh_steps):
            score_x = model(x, batch_t)
            noise = torch.randn_like(x)
            y = x + h_t * score_x + torch.sqrt(2 * h_t) * noise

            score_y = model(y, batch_t)
            drift = h_t * (score_x + score_y)
            log_q = -(0.25 / h_t) * torch.sum(
                (torch.sqrt(2 * h_t) * noise + drift) ** 2, dim=(1, 2, 3)
            ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3))
            log_pi = action_fn(x) - action_fn(y)

            accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
            accept = torch.rand(num_samples, device=device) < accept_prob
            x = torch.where(accept[:, None, None, None], y, x)
            total_accept += accept.float()

    acc = total_accept / mh_steps
    return acc.mean().item(), (acc.std() / acc.shape[0] ** 0.5).item()


def find_data_path(data_dir, k, l, L):
    """Find training data file, trying common naming conventions."""
    candidates = [
        data_dir / f"cfgs_wolff_fahmc_k={k}_l={l}_{L}^2.jld2",
        data_dir / f"cfgs_wolff_fahmc_k={k}_l=0.022_{L}^2.jld2",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fuzzy match: find any file matching k and L
    for p in sorted(data_dir.glob(f"cfgs_wolff_fahmc_k=*_l=*_{L}^2.jld2")):
        k_m = re.search(r'k=([\d.]+)', p.name)
        if k_m and abs(float(k_m.group(1)) - k) < 1e-6:
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=20)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--epoch_lo", type=int, default=3999)
    parser.add_argument("--epoch_hi", type=int, default=4999)
    parser.add_argument("--data_dir", type=str, default="trainingdata")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_dir = Path(".")

    # Auto-discover all (L, k) model directories
    pattern = f"phi4_L*_k*_l{args.l}_{args.network}"
    model_dirs = sorted(base_dir.glob(pattern))
    if not model_dirs:
        print(f"No directories matching {pattern}")
        return

    tasks = []
    for d in model_dirs:
        m_L = re.search(r'_L(\d+)', d.name)
        m_k = re.search(r'_k([\d.]+)', d.name)
        if m_L and m_k:
            tasks.append((int(m_L.group(1)), float(m_k.group(1)), d))
    tasks.sort(key=lambda x: (x[0], x[1]))

    print(f"Found {len(tasks)} (L, k) combinations:")
    for L, k, d in tasks:
        print(f"  L={L:4d}  k={k}")
    print(f"\nEpoch range: [{args.epoch_lo}, {args.epoch_hi}]")
    print(f"Parameters:  c={args.c}, t_mh={args.t_mh:.1e}\n")

    np.random.seed(42)

    results = []  # (L, k, acc_mean, acc_err, n_epochs)

    for L, k, model_dir in tasks:
        print(f"{'='*55}")
        print(f"L={L}, k={k}")
        print(f"{'='*55}")

        ckpts = find_epoch_checkpoints(model_dir / "models")
        target_epochs = sorted(
            e for e in ckpts if args.epoch_lo <= e <= args.epoch_hi
        )
        if not target_epochs:
            print(f"  WARNING: no checkpoints in [{args.epoch_lo}, {args.epoch_hi}], skipping")
            results.append((L, k, float('nan'), float('nan'), 0))
            continue
        print(f"  Checkpoints in range: {len(target_epochs)} "
              f"(epochs {target_epochs[0]}..{target_epochs[-1]})")

        # Load norm params from first target checkpoint
        ref_ckpt = torch.load(ckpts[target_epochs[0]], map_location="cpu",
                               weights_only=False)
        hparams = ref_ckpt.get("hyper_parameters", {})
        norm_min = hparams.get("norm_min") or -6.22
        norm_max = hparams.get("norm_max") or 6.19
        del ref_ckpt

        # Find training data
        data_path = find_data_path(data_dir, k, args.l, L)
        if data_path is None:
            print(f"  WARNING: no training data found for k={k}, L={L}, skipping")
            results.append((L, k, float('nan'), float('nan'), 0))
            continue
        print(f"  Data: {data_path.name}")

        x_ref = load_reference_data(
            data_path, args.num_samples, norm_min, norm_max, args.device
        )

        h = args.c / (L ** 2)
        epoch_accs = []

        for ep in target_epochs:
            model, action_fn, _, _ = load_model(ckpts[ep], args.device)
            mean, _ = measure_acceptance(model, action_fn, x_ref, h,
                                         args.t_mh, args.mh_steps)
            epoch_accs.append(mean)
            print(f"    epoch {ep:5d}  accept = {mean:.4f}")
            del model
            torch.cuda.empty_cache()

        avg = np.mean(epoch_accs)
        err = np.std(epoch_accs)
        print(f"  --> mean = {avg:.4f} ± {err:.4f}  (over {len(epoch_accs)} epochs)")
        results.append((L, k, avg, err, len(epoch_accs)))

    # Save CSV
    if args.output is None:
        out_csv = os.path.join(SCRIPT_DIR, f"acceptance_scan_ep{args.epoch_lo}-{args.epoch_hi}.csv")
    else:
        out_csv = args.output

    with open(out_csv, 'w') as f:
        f.write(f"# c={args.c}, t_mh={args.t_mh}, epoch_lo={args.epoch_lo}, "
                f"epoch_hi={args.epoch_hi}\n")
        f.write("L,k,acc_mean,acc_std,n_epochs\n")
        for L, k, acc, err, n in results:
            f.write(f"{L},{k},{acc:.6f},{err:.6f},{n}\n")

    print(f"\n{'='*55}")
    print(f"{'L':>6s}  {'k':>8s}  {'acc_mean':>10s}  {'acc_std':>10s}")
    print(f"{'='*55}")
    for L, k, acc, err, n in results:
        print(f"{L:6d}  {k:8.4f}  {acc:10.4f}  {err:10.4f}")
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
