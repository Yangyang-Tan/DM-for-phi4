"""
Convert cached propagator .npz → plain .dat (columns: |k|  G  G_err).

This is the format the Mathematica plotting script expects (matches the
celeba_128_ncsnpp/correlation/G_k_*.dat convention: tab/whitespace-separated
ASCII with 3 columns).

Output layout:
  results/sigma_comparison_L128/L128_k{k}_sigma{σ}/dat_cache/
      G_k_train.dat
      G_k_DM_em_epoch=NNNN.dat
      G_k_DM_ode_epoch=NNNN.dat
"""
import os, glob
import numpy as np

ROOT = "/data/tyywork/DM/2Dphi4/results/sigma_comparison_L128"

CFG = [
    ("0.2705", "sigma450"),
    ("0.28",   "sigma640"),
]

for k, sig in CFG:
    cache = f"{ROOT}/L128_k{k}_{sig}/prop_cache"
    out   = f"{ROOT}/L128_k{k}_{sig}/dat_cache"
    os.makedirs(out, exist_ok=True)
    for src in sorted(glob.glob(f"{cache}/*.npz")):
        name = os.path.basename(src)[:-4]                # strip .npz
        d = np.load(src)
        kv, G, Ge = d["k"], d["G"], d["Ge"]
        # layout columns (k, G, G_err), float precision
        arr = np.column_stack([kv, G, Ge])
        if name == "train":
            dst = f"{out}/G_k_train.dat"
        else:
            # em_ep0001 -> G_k_DM_em_epoch=0001.dat
            method, ep_token = name.split("_ep")   # ("em", "0001")
            dst = f"{out}/G_k_DM_{method}_epoch={ep_token}.dat"
        np.savetxt(dst, arr, fmt="%.10e", delimiter="\t")
    print(f"[{k} σ={sig}] wrote {len(os.listdir(out))} files -> {out}")
