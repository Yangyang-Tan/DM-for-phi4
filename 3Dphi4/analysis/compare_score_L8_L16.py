"""
Direct score comparison: L=8-only vs multi-L 4-8 (on L=8 input),
and L=16-only vs multi-L 4-8-16 (on L=16 input).

Method: take HMC cfg + identical noise realization, run through each model,
compare score outputs in real space and Fourier space.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import functools
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std

DEVICE = "cuda:2"
T_LIST  = [0.05, 0.1, 0.3, 0.5, 0.8]   # 5 t-values: UV → IR
N_CFG   = 64     # number of HMC cfgs to average over
SEED    = 42


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sigma = ckpt["hyper_parameters"]["sigma"]
    nm = ckpt["hyper_parameters"]["norm_min"]
    nM = ckpt["hyper_parameters"]["norm_max"]
    score_model = NCSNpp3D(functools.partial(marginal_prob_std, sigma=sigma))
    model = DiffusionModel3D.load_from_checkpoint(ckpt_path, score_model=score_model)
    return model.to(DEVICE).eval(), sigma, nm, nM


def load_hmc_normalized(L, k, lam, n, norm_min, norm_max):
    with h5py.File(f"trainingdata/cfgs_wolff_fahmc_k={k}_l={lam}_{L}^3.jld2", "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float32)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    cfgs = cfgs[:n]
    nrm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    return torch.from_numpy(nrm).unsqueeze(1).float()  # (n, 1, L, L, L)


@torch.no_grad()
def get_score(model, x_t, t):
    """Run model.forward (raw network score)."""
    with model.ema.average_parameters():
        batch_t = torch.full((x_t.shape[0],), t, device=x_t.device)
        return model(x_t, batch_t)


def diagonal_spectrum_3d(arr):
    """|FFT(arr)|² on diagonal modes (n,n,n) for n=1..L/2.
    arr: (N, L, L, L) numpy."""
    L = arr.shape[1]
    fk = np.fft.fftn(arr, axes=(1, 2, 3))
    g = (np.abs(fk) ** 2)
    diag_n = np.arange(1, L // 2 + 1)
    g_diag = g[:, diag_n, diag_n, diag_n]
    k_lat = np.sqrt(3.0) * 2.0 * np.sin(np.pi * diag_n / L)
    return k_lat, g_diag.mean(axis=0), g_diag.std(axis=0) / np.sqrt(arr.shape[0])


def compare_pair(model_a, model_b, hmc, sigma_a, sigma_b, t, label_a, label_b, ax_real, ax_fft):
    # add identical noise (use sigma_a for noise level — both models should learn similar score there)
    std_t_a = float(np.sqrt((sigma_a**(2*t) - 1) / (2*np.log(sigma_a))))
    std_t_b = float(np.sqrt((sigma_b**(2*t) - 1) / (2*np.log(sigma_b))))
    print(f"  {label_a}: σ={sigma_a:.0f}, std(t={t})={std_t_a:.2f}")
    print(f"  {label_b}: σ={sigma_b:.0f}, std(t={t})={std_t_b:.2f}")

    rng = torch.Generator(device=DEVICE).manual_seed(SEED)
    eps = torch.randn(hmc.shape, generator=rng, device=DEVICE)
    hmc_dev = hmc.to(DEVICE)

    # Each model needs noise scaled by its own std(t) (otherwise its loss target differs)
    x_a = hmc_dev + eps * std_t_a
    x_b = hmc_dev + eps * std_t_b
    score_a = get_score(model_a, x_a, t).cpu().numpy().astype(np.float64)[:, 0]
    score_b = get_score(model_b, x_b, t).cpu().numpy().astype(np.float64)[:, 0]

    # Real-space residual after rescaling (since each is roughly -eps/std)
    # Multiply by std_t to recover -eps prediction (target ≈ -eps), comparable across σ
    pred_a = score_a * std_t_a
    pred_b = score_b * std_t_b
    eps_np = eps.cpu().numpy()[:, 0]

    # Fourier spectra of the two predictions (should both ≈ -FFT(eps), but may differ)
    k_a, S_a, E_a = diagonal_spectrum_3d(pred_a)
    k_b, S_b, E_b = diagonal_spectrum_3d(pred_b)
    k_t, S_t, E_t = diagonal_spectrum_3d(-eps_np)  # ground-truth "score × std" target

    ax_fft.errorbar(k_a, S_a, yerr=E_a, fmt='o--', label=f"{label_a}", ms=4, lw=1.3)
    ax_fft.errorbar(k_b, S_b, yerr=E_b, fmt='s-',  label=f"{label_b}", ms=4, lw=1.3)
    ax_fft.errorbar(k_t, S_t, yerr=E_t, fmt='x:', label="target = $-\\epsilon$", ms=5, lw=1, color='black')

    # Spectrum of the per-voxel difference predicted vs target
    k_da, S_da, _ = diagonal_spectrum_3d(pred_a - (-eps_np))
    k_db, S_db, _ = diagonal_spectrum_3d(pred_b - (-eps_np))
    ax_real.plot(k_da, S_da, 'o--', label=f"{label_a} − target", lw=1.3, ms=4)
    ax_real.plot(k_db, S_db, 's-',  label=f"{label_b} − target", lw=1.3, ms=4)


@torch.no_grad()
def total_residual_norm(model, hmc_dev, sigma, t, eps):
    std_t = float(np.sqrt((sigma**(2*t) - 1) / (2*np.log(sigma))))
    x_t = hmc_dev + eps * std_t
    score = get_score(model, x_t, t).cpu().numpy().astype(np.float64)[:, 0]
    pred = score * std_t       # should ≈ -eps
    target = -eps.cpu().numpy()[:, 0]
    residual = pred - target   # the per-voxel score-matching error vector
    # mean per-voxel squared error (= score-matching loss / #voxels)
    return float((residual**2).mean()), residual


def main():
    print("Loading 4 models...")
    m8_only,   sig8o,   _, _ = load_model("runs/phi4_3d_L8_k0.1923_l0.9_ncsnpp_sigma100/models/epoch=10000.ckpt")
    m48,       sig48,   _, _ = load_model("runs/phi4_3d_Lmulti4-8_k0.1923_l0.9_ncsnpp/models/epoch=10000.ckpt")
    m16_only,  sig16o,  nm, nM = load_model("runs/phi4_3d_L16_k0.1923_l0.9_ncsnpp_sigma280/models/epoch=10000.ckpt")
    m4816,     sig4816, _, _ = load_model("runs/phi4_3d_Lmulti4-8-16_k0.1923_l0.9_ncsnpp/models/epoch=10000.ckpt")

    _, _, nm8, nM8 = load_model("runs/phi4_3d_L8_k0.1923_l0.9_ncsnpp_sigma100/models/epoch=10000.ckpt")
    hmc8  = load_hmc_normalized(8,  0.1923, 0.9, N_CFG, nm8, nM8).to(DEVICE)
    hmc16 = load_hmc_normalized(16, 0.1923, 0.9, N_CFG, nm,  nM ).to(DEVICE)

    # Also load HMC L=64 to test cross-L score
    hmc64 = load_hmc_normalized(64, 0.1923, 0.9, 8, nm, nM).to(DEVICE)  # tiny N for memory
    rng = torch.Generator(device=DEVICE).manual_seed(SEED)
    eps8  = torch.randn(hmc8.shape,  generator=rng, device=DEVICE)
    eps16 = torch.randn(hmc16.shape, generator=rng, device=DEVICE)
    eps64 = torch.randn(hmc64.shape, generator=rng, device=DEVICE)
    # Override model.L for L=64 evaluation (otherwise sample shape mismatch unused)
    for m in [m8_only, m48, m16_only, m4816]: m.L = 64

    print()
    print("--- CROSS-L test: HMC L=64 input → each model's score residual ---")
    print(f"{'t':>5}  {'L=8-only':>14}  {'multi-L 4-8':>14}  {'L=16-only':>14}  {'multi-L 4-8-16':>16}")
    for t in T_LIST:
        l8o,_  = total_residual_norm(m8_only,  hmc64, sig8o,  t, eps64)
        l48,_  = total_residual_norm(m48,      hmc64, sig48,  t, eps64)
        l16o,_ = total_residual_norm(m16_only, hmc64, sig16o, t, eps64)
        l4816,_= total_residual_norm(m4816,    hmc64, sig4816,t, eps64)
        print(f"{t:>5.2f}  {l8o:>14.5f}  {l48:>14.5f}  {l16o:>14.5f}  {l4816:>16.5f}")

    print()
    print(f"{'t':>5}  {'L=8-only':>14}  {'multi-L 4-8':>14}  {'ratio':>7}  ||  "
          f"{'L=16-only':>14}  {'multi-L 4-8-16':>16}  {'ratio':>7}")
    print("-" * 110)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    series = {k: {} for k in ["L=8-only", "multi-L 4-8", "L=16-only", "multi-L 4-8-16"]}
    for t in T_LIST:
        loss_8o, _ = total_residual_norm(m8_only,  hmc8,  sig8o,  t, eps8)
        loss_48, _ = total_residual_norm(m48,      hmc8,  sig48,  t, eps8)
        loss_16o,_ = total_residual_norm(m16_only, hmc16, sig16o, t, eps16)
        loss_4816,_= total_residual_norm(m4816,    hmc16, sig4816,t, eps16)
        print(f"{t:>5.2f}  {loss_8o:>14.5f}  {loss_48:>14.5f}  {loss_8o/loss_48:>7.2f}  ||  "
              f"{loss_16o:>14.5f}  {loss_4816:>16.5f}  {loss_16o/loss_4816:>7.2f}")
        series["L=8-only"][t]       = loss_8o
        series["multi-L 4-8"][t]    = loss_48
        series["L=16-only"][t]      = loss_16o
        series["multi-L 4-8-16"][t] = loss_4816

    # Plot residual norm vs t (the per-voxel score-matching residual squared)
    for ax, names, title in [
        (axes[0], ["L=8-only", "multi-L 4-8"],       "L=8 input — score residual (target = $-\\epsilon$)"),
        (axes[1], ["L=16-only", "multi-L 4-8-16"], "L=16 input — score residual"),
    ]:
        for n in names:
            ax.plot(list(series[n].keys()), list(series[n].values()),
                    'o-', label=n, lw=1.5, ms=6)
        ax.set_xlabel("t  (forward-SDE time)")
        ax.set_ylabel(r"$\langle(s_\theta\cdot\sigma+\epsilon)^2\rangle$  per voxel")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=10)
        ax.invert_xaxis()
    fig.suptitle(f"Score-matching residual at fixed t (per-voxel, N={N_CFG} HMC cfgs, κ=0.1923)",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    out = Path("results/score_residual_vs_t.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
