"""
3D wrapper around the unified ``FieldDataModule``.

Kept for backwards compatibility. Defaults differ from 2D:
  * ``cache_norm=True`` (3D loads are slow enough that caching is worth it),
  * ``num_workers=4`` (matches the original ``FieldDataModule3D``).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytorch_lightning as pl

from data import (
    FieldDataModule, GPUDataLoader, MultiLBatchSampler,
    _normalize_pm1, _renorm_pm1,
)  # noqa: F401  (some re-exported)


# Back-compat alias for callers that imported ``GPUDataLoader3D`` directly.
GPUDataLoader3D = GPUDataLoader


class FieldDataModule3D(FieldDataModule):
    def __init__(self, data_path, batch_size=64, normalize=True,
                 num_workers=4, device=None, cache_norm=True):
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            normalize=normalize,
            device=device,
            num_workers=num_workers,
            cache_norm=cache_norm,
        )


class MultiLFieldDataModule3D(pl.LightningDataModule):
    """3D multi-lattice-size DataModule. Mirrors the 2D MultiLFieldDataModule
    but accepts 4-D cfgs (N, L, L, L) and auto-detects (L, L, L, N) layouts.

    Pooled global [-1, 1] normalisation; per-step single-L batching via
    MultiLBatchSampler.
    """

    def __init__(self, data_paths, batch_size=64, normalize=True, device=None):
        super().__init__()
        self.data_paths = list(data_paths)
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device

        self.cfgs_min = None
        self.cfgs_max = None
        self.data_by_L = None    # dict: L -> tensor (N, 1, L, L, L)
        self.Ls = None
        self.N_total = 0

    def setup(self, stage=None):
        import h5py
        self.N_total = 0
        raws = {}
        for path in self.data_paths:
            with h5py.File(path, "r") as f:
                cfgs = np.array(f["cfgs"], dtype=np.float32)
            assert cfgs.ndim == 4, (
                f"3D module expects ndim=4 (N,L,L,L); got {cfgs.shape} in {path}"
            )
            # Auto-detect (L, L, L, N) -> (N, L, L, L)
            if cfgs.shape[-1] > cfgs.shape[0]:
                cfgs = np.ascontiguousarray(cfgs.transpose(3, 0, 1, 2))
            L = int(cfgs.shape[1])
            assert cfgs.shape[2] == L and cfgs.shape[3] == L, (
                f"non-cubic lattice in {path}: {cfgs.shape}"
            )
            raws[L] = cfgs

        if self.normalize:
            self.cfgs_min = float(min(r.min() for r in raws.values()))
            self.cfgs_max = float(max(r.max() for r in raws.values()))
            print(f"Multi-L pooled norm: [{self.cfgs_min:.4f}, {self.cfgs_max:.4f}] "
                  f"over Ls = {sorted(raws.keys())}")
            for L in raws:
                raws[L] = _normalize_pm1(raws[L], self.cfgs_min, self.cfgs_max)

        self.data_by_L = {}
        for L, arr in raws.items():
            t = torch.from_numpy(arr).unsqueeze(1).float()
            if self.device:
                t = t.to(self.device)
            self.data_by_L[L] = t
            self.N_total += t.shape[0]
        self.Ls = sorted(self.data_by_L.keys())

        sizes = ", ".join(f"L={L}: N={self.data_by_L[L].shape[0]}" for L in self.Ls)
        if self.device:
            total_gb = sum(t.nbytes for t in self.data_by_L.values()) / 1e9
            print(f"Multi-L 3D data on {self.device}: {sizes}  ({total_gb:.2f} GB)")
        else:
            print(f"Multi-L 3D data on CPU: {sizes}")

    def train_dataloader(self):
        return MultiLBatchSampler(self.data_by_L, self.batch_size)

    def renorm(self, y):
        if self.cfgs_min is None:
            return y
        return _renorm_pm1(y, self.cfgs_min, self.cfgs_max)
