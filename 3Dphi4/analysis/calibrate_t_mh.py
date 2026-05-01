"""
Calibrate t_mh for 3D phi4 by sweeping t_mh on early vs late checkpoint.

Usage:
    python calibrate_t_mh.py --device cuda:2 --L 64 --k 0.2 --c 0.2
"""

import sys
import re, io, functools, argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))   # 3Dphi4/
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # repo root

import h5py, torch, numpy as np, matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from sample_phi4 import phi4_action_3d


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
    score_model = NCSNpp3D(marginal_prob_std_fn)
    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf); del ckpt; buf.seek(0)
    model = DiffusionModel3D.load_from_checkpoint(buf, score_model=score_model)
    model = model.to(device).eval()
    action_fn = functools.partial(phi4_action_3d, k=k, l=l, phi_min=norm_min, phi_max=norm_max)
    return model, action_fn


def load_reference_data(data_path, num_samples, norm_min, norm_max, device):
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    idx = np.random.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    return x


def sweep_t_mh(model, action_fn, h, t_mh_values, x_ref, mh_steps, batch_size=128):
    results = []
    device = model.device
    num_samples = x_ref.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x_ref.dtype)
    for t_val in t_mh_values:
        x = x_ref.clone()
        total_accept = torch.zeros(num_samples, device=device)
        with torch.no_grad():
            for _ in range(mh_steps):
                score_x = torch.cat([model(x[i:i+batch_size], torch.ones(min(batch_size, num_samples-i), device=device)*t_val) for i in range(0, num_samples, batch_size)])
                noise = torch.randn_like(x)
                y = x + h_t * score_x + torch.sqrt(2 * h_t) * noise
                score_y = torch.cat([model(y[i:i+batch_size], torch.ones(min(batch_size, num_samples-i), device=device)*t_val) for i in range(0, num_samples, batch_size)])
                drift = h_t * (score_x + score_y)
                log_q = -(0.25 / h_t) * torch.sum(
                    (torch.sqrt(2 * h_t) * noise + drift) ** 2, dim=(1, 2, 3, 4)
                ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3, 4))
                log_pi = action_fn(x) - action_fn(y)
                accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
                accept = torch.rand(num_samples, device=device) < accept_prob
                x = torch.where(accept[:, None, None, None, None], y, x)
                total_accept += accept.float()
        acc = total_accept / mh_steps
        results.append((acc.mean().item(), (acc.std() / acc.shape[0] ** 0.5).item()))
        print(f"    t_mh = {t_val:.1e}  ->  accept = {acc.mean().item():.4f} +- {acc.std().item():.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--network", type=str, default="ncsnpp")
    parser.add_argument("--early_epoch", type=int, default=None)
    parser.add_argument("--late_epoch", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--mh_steps", type=int, default=1)
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    model_dir = Path(f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_{args.network}")
    ckpts = find_epoch_checkpoints(model_dir / "models")
    epochs = sorted(ckpts.keys())
    early_ep = args.early_epoch or epochs[0]
    late_ep = args.late_epoch or epochs[-1]
    print(f"Early checkpoint: epoch {early_ep}")
    print(f"Late  checkpoint: epoch {late_ep}")

    ref_ckpt = torch.load(ckpts[early_ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    del ref_ckpt
    print(f"norm range: [{norm_min}, {norm_max}]")

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"
    np.random.seed(42)
    x_ref = load_reference_data(args.data_path, args.num_samples, norm_min, norm_max, args.device)
    print(f"Loaded {x_ref.shape[0]} reference configs, shape {x_ref.shape}")

    L = args.L
    h = args.c / (L ** 3)
    t_mh_values = np.array([1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1])
    print(f"\nFixed step size: c = {args.c}, h = {h:.2e}")
    print(f"Sweeping t_mh over {t_mh_values.tolist()}\n")

    print(f"{'='*50}\nEarly model  (epoch {early_ep})\n{'='*50}")
    model_early, action_fn = load_model(ckpts[early_ep], args.device)
    res_early = sweep_t_mh(model_early, action_fn, h, t_mh_values, x_ref, args.mh_steps)
    del model_early; torch.cuda.empty_cache()

    print(f"\n{'='*50}\nLate model   (epoch {late_ep})\n{'='*50}")
    model_late, action_fn = load_model(ckpts[late_ep], args.device)
    res_late = sweep_t_mh(model_late, action_fn, h, t_mh_values, x_ref, args.mh_steps)
    del model_late; torch.cuda.empty_cache()

    acc_early = np.array([r[0] for r in res_early])
    acc_late = np.array([r[0] for r in res_late])
    gap = acc_late - acc_early

    print(f"\n{'='*60}")
    print(f"{'t_mh':>10s}  {'early':>7s}  {'late':>7s}  {'gap':>7s}")
    print(f"{'='*60}")
    for i, t in enumerate(t_mh_values):
        print(f"{t:10.1e}  {acc_early[i]:7.4f}  {acc_late[i]:7.4f}  {gap[i]:+7.4f}")

    useful = (acc_late > 0.3) & (acc_late < 0.85)
    best_idx = np.where(useful, gap, -np.inf).argmax() if useful.any() else gap.argmax()
    print(f"\nRecommended:  t_mh = {t_mh_values[best_idx]:.1e}")
    print(f"  early accept = {acc_early[best_idx]:.4f}")
    print(f"  late  accept = {acc_late[best_idx]:.4f}")
    print(f"  gap          = {gap[best_idx]:+.4f}")

    # --- plot ---
    def _sci(v):
        e = int(np.floor(np.log10(v)))
        m = v / 10**e
        return rf'10^{{{e}}}' if m == 1 else rf'{m:.0f}\!\times\!10^{{{e}}}'

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(t_mh_values, acc_early, yerr=[r[1] for r in res_early],
                 label=f'Epoch {early_ep} (early)', marker='o', capsize=3, color='C3')
    ax1.errorbar(t_mh_values, acc_late, yerr=[r[1] for r in res_late],
                 label=f'Epoch {late_ep} (late)', marker='s', capsize=3, color='C0')
    ax1.axvline(t_mh_values[best_idx], ls='--', color='gray', alpha=0.6,
                label=rf'Recommended $t_{{\mathrm{{mh}}}}={_sci(t_mh_values[best_idx])}$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$t_{\mathrm{mh}}$')
    ax1.set_ylabel('Acceptance rate')
    ax1.set_title(rf'3D: $L={L},\;\kappa={args.k},\;c={args.c}$')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    bar_labels = []
    for t in t_mh_values:
        e = int(np.floor(np.log10(t)))
        m = t / 10**e
        bar_labels.append(rf'$10^{{{e}}}$' if m == 1 else rf'${m:.0f}\!\times\!10^{{{e}}}$')
    colors = ['C2' if useful[i] else 'C7' for i in range(len(t_mh_values))]
    ax2.bar(range(len(t_mh_values)), gap, tick_label=bar_labels, color=colors)
    ax2.set_xlabel(r'$t_{\mathrm{mh}}$')
    ax2.set_ylabel('Acceptance gap (late $-$ early)')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.grid(alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), fontsize=9)

    fig.tight_layout()
    out_path = f"calibrate_t_mh_3d_L{L}_k{args.k}.pdf"
    fig.savefig(out_path)
    print(f"\nSaved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
