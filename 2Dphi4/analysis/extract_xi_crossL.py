"""
Extract correlation length ξ from the cross-L momentum-space propagator.

Fit  G(k̂²) = Z / (k̂² + m²)  on the lowest few radial bins.
ξ = 1/m.  Bootstrap over configurations to get an error on ξ.

This tests the prediction that ξ_DM is capped near L_train / 2 ≈ 16
because L=32 training data has no longer-range structure.
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
    sample_axis = int(np.argmax(cfgs.shape))
    if sample_axis != 0:
        cfgs = np.moveaxis(cfgs, sample_axis, 0)
    return cfgs


def _signed_n(L: int) -> np.ndarray:
    n = np.arange(L)
    return np.where(n > L // 2, n - L, n)


def lattice_kh2_grid(L: int) -> tuple[np.ndarray, np.ndarray]:
    n1 = _signed_n(L)
    p = 4 * np.sin(np.pi * n1 / L) ** 2
    KH2 = p[:, None] + p[None, :]
    sum_p4 = (p[:, None]) ** 2 + (p[None, :]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = np.where(KH2 > 1e-12, sum_p4 / KH2 ** 2, 0.0)
    return KH2, diag


def per_config_pk(cfgs: np.ndarray) -> np.ndarray:
    L = cfgs.shape[1]
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    return (fk * fk.conj()).real / V


def radial_pk_per_config(values_per_cfg: np.ndarray, KH2: np.ndarray,
                         mask: np.ndarray, n_bins: int):
    """Return (k_centres, per-config-radial values shape (N, nbin)) using log-spaced bins."""
    k_sel = KH2[mask]
    pos = k_sel[k_sel > 1e-10]
    edges = np.geomspace(pos.min(), pos.max() * 1.001, n_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    N = values_per_cfg.shape[0]
    out = np.zeros((N, n_bins))
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        in_bin = (k_sel >= edges[b]) & (k_sel < edges[b + 1])
        if not np.any(in_bin):
            continue
        counts[b] = in_bin.sum()
        out[:, b] = values_per_cfg[:, mask][:, in_bin].mean(axis=1)
    valid = counts > 0
    return centres[valid], out[:, valid]


def fit_xi(k2_centres: np.ndarray, G_per_cfg: np.ndarray,
           n_low: int = 5, n_boot: int = 200, seed: int = 0):
    """Fit  1/G = (k² + m²) / Z  using the lowest n_low radial bins (linear in k²)."""
    rng = np.random.default_rng(seed)
    N = G_per_cfg.shape[0]
    k2 = k2_centres[:n_low]

    def _fit_once(G_mean):
        invG = 1.0 / G_mean
        # invG = a*k2 + b   →  Z = 1/a, m² = b/a
        a, b = np.polyfit(k2, invG, 1)
        if a <= 0 or b <= 0:
            return np.nan, np.nan, np.nan
        Z = 1.0 / a
        m2 = b / a
        return Z, m2, 1.0 / np.sqrt(m2)

    Z0, m20, xi0 = _fit_once(G_per_cfg.mean(axis=0)[:n_low])
    xi_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        G_b = G_per_cfg[idx].mean(axis=0)[:n_low]
        try:
            _, _, xi_boot[b] = _fit_once(G_b)
        except Exception:
            xi_boot[b] = np.nan
    return Z0, m20, xi0, np.nanstd(xi_boot)


def main():
    L = 64
    n_bins = 14
    n_low = 5
    KH2, DIAG = lattice_kh2_grid(L)
    mask = (DIAG <= 0.51) & (KH2 > 1e-10)

    cases = [("0.26", "4999", "symmetric"),
             ("0.2705", "9999", "near-critical"),
             ("0.28", "4999", "broken")]

    print(f"{'phase':>14}  {'κ':>7}   {'source':>14}    {'Z':>7}  {'m²':>7}    {'ξ':>10}    {'L_train/2':>9}")
    print("-" * 90)

    rows = []
    for k, ep, label in cases:
        ref_path = Path(f"trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_64^2.jld2")
        ref = load_hmc(ref_path)
        if ref.shape[0] > 8192:
            ref = ref[:8192]
        kc, Gpc = radial_pk_per_config(per_config_pk(ref), KH2, mask, n_bins)
        Z, m2, xi, xi_err = fit_xi(kc, Gpc, n_low=n_low)
        print(f"{label:>14}  {k:>7}   {'HMC L=64':>14}    {Z:7.3f}  {m2:7.4f}    {xi:6.2f}±{xi_err:4.2f}    {16:>9}")
        rows.append((label, k, "HMC", xi, xi_err))

        for sched in ("log", "linear"):
            f = Path(f"runs/phi4_L32_k{k}_l0.022_ncsnpp/data_crossL/"
                     f"samples_crossL_train32_sample64_em_{sched}_steps2000_ep{ep}.npy")
            if not f.exists():
                continue
            d = load_dm(f)
            kc_d, Gpc_d = radial_pk_per_config(per_config_pk(d), KH2, mask, n_bins)
            Z, m2, xi, xi_err = fit_xi(kc_d, Gpc_d, n_low=n_low)
            print(f"{label:>14}  {k:>7}   {'DM[' + sched + ']':>14}    {Z:7.3f}  {m2:7.4f}    {xi:6.2f}±{xi_err:4.2f}    {16:>9}")
            rows.append((label, k, f"DM[{sched}]", xi, xi_err))
        print("")

    out = Path("results/crossL/xi_table.txt")
    with out.open("w") as f:
        f.write("phase  kappa  source  xi  xi_err\n")
        for r in rows:
            f.write(f"{r[0]}  {r[1]}  {r[2]}  {r[3]:.4f}  {r[4]:.4f}\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
