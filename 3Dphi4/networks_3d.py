"""
3D network wrappers for score-based diffusion models.

The implementations live in ``../networks_nd.py`` and are shared with the 2D
architectures.  This module preserves the old class names used by 3D scripts
and checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from networks_nd import (  # noqa: E402
    AttnBlockND,
    DenseND,
    GaussianFourierProjection,
    NCSNppND,
    ResnetBlockND,
    ScoreNetND,
    ScoreNetPeriodicUNetND,
)


class Dense3D(DenseND):
    """A fully connected layer that reshapes outputs to 3D feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, spatial_dims=3)


class ScoreNet3DUNetPeriodic(ScoreNetPeriodicUNetND):
    """3D periodic U-Net with downsampling/upsampling."""

    def __init__(self, marginal_prob_std_fn, channels=[32, 64, 128, 256],
                 embed_dim=256):
        super().__init__(marginal_prob_std_fn, spatial_dims=3,
                         channels=channels, embed_dim=embed_dim)


class ScoreNet3D(ScoreNetND):
    """3D no-downsampling score model."""

    def __init__(self, marginal_prob_std_fn, channels=[32, 64, 128, 256],
                 embed_dim=256, periodic: bool = False):
        super().__init__(
            marginal_prob_std_fn,
            spatial_dims=3,
            channels=channels,
            embed_dim=embed_dim,
            periodic=periodic,
        )


class ResnetBlock3D(ResnetBlockND):
    """3D ResNet block with additive time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__(in_ch, out_ch, embed_dim, spatial_dims=3,
                         periodic=True)


class AttnBlock3D(AttnBlockND):
    """Self-attention block for 3D data."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__(channels, spatial_dims=3, num_heads=num_heads)


class NCSNpp3D(NCSNppND):
    """NCSN++ style 3D U-Net with ResNet blocks and time conditioning."""

    def __init__(self, marginal_prob_std_fn, channels=[10, 20, 40, 80],
                 embed_dim=256, use_attention=False):
        super().__init__(
            marginal_prob_std_fn,
            spatial_dims=3,
            channels=channels,
            embed_dim=embed_dim,
            use_attention=use_attention,
            periodic=True,
        )


NCSNpp3DSimple = NCSNpp3D


__all__ = [
    "GaussianFourierProjection",
    "Dense3D",
    "ScoreNet3D",
    "ScoreNet3DUNetPeriodic",
    "ResnetBlock3D",
    "AttnBlock3D",
    "NCSNpp3D",
    "NCSNpp3DSimple",
]
