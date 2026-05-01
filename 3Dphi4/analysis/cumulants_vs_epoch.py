"""Compute cumulants kappa_{2,4,6,8} vs training epoch for 3D phi^4 L=64 k=0.2.

Reference is the HMC (Wolff+FAHMC) training dataset; DM samples are read from
``samples_epoch=*.npy`` under the checkpoint's ``data/`` dir. Both are Z2
symmetrised (odd moments zeroed) before cumulants are formed from moments.

Outputs under ``<ckpt>/cumulants/``:
  cumulants_hmc.dat            (order kappa err)
  cumulants_dm_em.dat          (epoch k2 k2e k4 k4e k6 k6e k8 k8e)
  hmc_per_sample_moments.npy   (cache of per-sample moments)
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.special import binom


def cumulants_from_moments(m):
    order = m.shape[0]
    kappa = np.empty_like(m, dtype=np.float64)
    kappa[0] = m[0]
    for n in range(2, order + 1):
        v = m[n - 1].copy()
        for i in range(1, n):
            v -= binom(n - 1, i - 1) * kappa[i - 1] * m[n - i - 1]
        kappa[n - 1] = v
    return kappa


def per_sample_moments(batch_iter, order=8):
    """Return (N, order) array of spatially-averaged <phi^k> per sample."""
    out = []
    for batch in batch_iter:
        x = batch.reshape(batch.shape[0], -1).astype(np.float64, copy=False)
        m = np.empty((x.shape[0], order), dtype=np.float64)
        cur = np.ones_like(x)
        for k in range(order):
            cur = cur * x
            m[:, k] = cur.mean(axis=1)
        out.append(m)
    return np.concatenate(out, axis=0)


def bootstrap_cumulants(per_sample_m, orders=(2, 4, 6, 8),
                        n_boot=400, n_bins=32, z2=True, seed=0):
    rng = np.random.default_rng(seed)
    N, order = per_sample_m.shape
    n_bins = int(min(n_bins, N))
    bs = N // n_bins
    pm = per_sample_m[: bs * n_bins]
    per_bin = pm.reshape(n_bins, bs, order).mean(axis=1)
    if z2:
        for k in range(1, order + 1):
            if k % 2 == 1:
                per_bin[:, k - 1] = 0.0

    boot = np.empty((n_boot, order), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_bins, size=n_bins)
        boot[b] = cumulants_from_moments(per_bin[idx].mean(axis=0))

    return {k: (float(boot[:, k - 1].mean()), float(boot[:, k - 1].std()))
            for k in orders}


def iter_h5(path, key="cfgs", chunk=64):
    with h5py.File(path, "r") as f:
        ds = f[key]
        N = ds.shape[0]
        for i in range(0, N, chunk):
            yield np.asarray(ds[i:i + chunk])


def iter_npy(path, chunk=64):
    arr = np.load(path)                    # (L, L, L, N)
    arr = np.moveaxis(arr, -1, 0)          # (N, L, L, L)
    N = arr.shape[0]
    for i in range(0, N, chunk):
        yield arr[i:i + chunk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="runs/phi4_3d_L64_k0.2_l0.9_ncsnpp")
    ap.add_argument("--hmc_path",
                    default="trainingdata/cfgs_wolff_fahmc_k=0.2_l=0.9_64^3.jld2")
    ap.add_argument("--data_subdir", default="data",
                    help="subdir under ckpt_dir holding sample .npy files")
    ap.add_argument("--sampler_tag", default="",
                    help="e.g. 'em' or 'ode' — filters files matching "
                         "'samples_{tag}_steps\\d+_epoch=NNNN.npy' and names "
                         "the output cumulants_dm_{tag}.dat. Empty = match "
                         "legacy 'samples_epoch=NNNN.npy' (output _em).")
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.ckpt_dir)
    data_dir = root / args.data_subdir
    outdir = root / "cumulants"
    outdir.mkdir(parents=True, exist_ok=True)
    orders = (2, 4, 6, 8)

    # -------- HMC reference --------
    hmc_cache = outdir / "hmc_per_sample_moments.npy"
    if hmc_cache.exists():
        print(f"[HMC] load cache {hmc_cache}")
        m_hmc = np.load(hmc_cache)
    else:
        print(f"[HMC] computing moments from {args.hmc_path}")
        m_hmc = per_sample_moments(iter_h5(args.hmc_path, chunk=args.chunk),
                                    order=args.order)
        np.save(hmc_cache, m_hmc)
        print(f"[HMC] cached {hmc_cache}  shape={m_hmc.shape}")
    hmc = bootstrap_cumulants(m_hmc, orders=orders, n_boot=args.n_boot,
                              n_bins=args.n_bins, z2=True, seed=args.seed)
    hmc_out = outdir / "cumulants_hmc.dat"
    with open(hmc_out, "w") as f:
        f.write("# order  kappa  err\n")
        for k in orders:
            mu, er = hmc[k]
            f.write(f"{k} {mu:.8e} {er:.8e}\n")
    print(f"[HMC] wrote {hmc_out}")
    for k in orders:
        print(f"       k{k} = {hmc[k][0]:.4g} ± {hmc[k][1]:.2g}")

    # -------- DM per epoch --------
    if args.sampler_tag:
        pat = re.compile(
            rf"samples_{re.escape(args.sampler_tag)}_steps\d+_epoch=(\d+)\.npy$")
        out_tag = args.sampler_tag
    else:
        pat = re.compile(r"samples_epoch=(\d+)\.npy$")
        out_tag = "em"
    by_epoch = {}
    for p in sorted(data_dir.iterdir()):
        m = pat.search(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        prev = by_epoch.get(ep)
        # Prefer larger file (more samples) when duplicated (e.g., 0599 vs 599)
        if prev is None or p.stat().st_size > prev.stat().st_size:
            by_epoch[ep] = p
    epochs = sorted(by_epoch.items())
    print(f"[DM] {len(epochs)} epochs found (pattern={pat.pattern})")

    dm_out = outdir / f"cumulants_dm_{out_tag}.dat"
    with open(dm_out, "w") as f:
        f.write("# epoch  k2 k2e  k4 k4e  k6 k6e  k8 k8e\n")
        for ep, p in epochs:
            m_dm = per_sample_moments(iter_npy(str(p), chunk=args.chunk),
                                       order=args.order)
            n_bins_ep = min(args.n_bins, m_dm.shape[0])
            res = bootstrap_cumulants(m_dm, orders=orders, n_boot=args.n_boot,
                                      n_bins=n_bins_ep, z2=True,
                                      seed=args.seed + ep)
            cols = []
            for k in orders:
                mu, er = res[k]
                cols.append(f"{mu:.8e} {er:.8e}")
            f.write(f"{ep} " + " ".join(cols) + "\n")
            f.flush()
            print(f"[DM] ep={ep:>5}  "
                  f"k2={res[2][0]:.3g}  k4={res[4][0]:.3g}  "
                  f"k6={res[6][0]:.3g}  k8={res[8][0]:.3g}")

    print(f"[DM] wrote {dm_out}")


if __name__ == "__main__":
    main()
