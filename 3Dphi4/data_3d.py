"""
3D wrapper around the unified ``FieldDataModule``.

Kept for backwards compatibility. Defaults differ from 2D:
  * ``cache_norm=True`` (3D loads are slow enough that caching is worth it),
  * ``num_workers=4`` (matches the original ``FieldDataModule3D``).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import FieldDataModule, GPUDataLoader  # noqa: F401  (re-exported)


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
