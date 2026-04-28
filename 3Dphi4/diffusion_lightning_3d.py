"""
3D wrapper around the unified ``DiffusionModel``.

Kept for backwards compatibility: existing 3D scripts (and existing 3D
checkpoints, whose saved hyperparameters do not include ``spatial_dims``)
continue to work via this thin subclass.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from diffusion_lightning import (  # noqa: F401  (re-exported)
    DiffusionModel,
    marginal_prob_std,
    diffusion_coeff,
)


class DiffusionModel3D(DiffusionModel):
    def __init__(self, *args, spatial_dims=3, **kwargs):
        super().__init__(*args, spatial_dims=spatial_dims, **kwargs)
