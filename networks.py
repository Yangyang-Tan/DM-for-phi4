"""
Neural network architectures for score-based diffusion models.

The lattice-field architectures are thin 2D wrappers around shared
dimension-agnostic implementations in ``networks_nd.py``.  This keeps existing
imports and checkpoint keys stable while avoiding separate 2D/3D code paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networks_nd import (
    AttnBlockND,
    DenseND,
    GaussianFourierProjection,
    NCSNppND,
    ResnetBlockND,
    ScoreNetND,
    ScoreNetPeriodicUNetND,
)


class Dense(DenseND):
    """A fully connected layer that reshapes outputs to 2D feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, spatial_dims=2)


class ScoreNetUNet(nn.Module):
    """U-Net score model with downsampling/upsampling for image data."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256],
                 embed_dim=256, in_channels=1, image_size=28):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1,
                               bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2,
                               bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2,
                               bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2,
                               bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        tconv4_output_padding = 1 if image_size >= 32 else 0
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False,
            output_padding=tconv4_output_padding,
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2,
            bias=False, output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2,
            bias=False, output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0],
                                         in_channels, 3, stride=1)

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1))
                                  + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1))
                                  + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        return h / self.marginal_prob_std(t)[:, None, None, None]


class ScoreNetUNetPeriodic(ScoreNetPeriodicUNetND):
    """2D periodic U-Net for lattice field theory."""

    def __init__(self, marginal_prob_std_fn, channels=[32, 64, 128, 256],
                 embed_dim=256):
        super().__init__(marginal_prob_std_fn, spatial_dims=2,
                         channels=channels, embed_dim=embed_dim)


class ScoreNet(ScoreNetND):
    """2D no-downsampling score model."""

    def __init__(self, marginal_prob_std_fn, channels=[32, 64, 128, 256],
                 embed_dim=256, periodic: bool = False):
        super().__init__(
            marginal_prob_std_fn,
            spatial_dims=2,
            channels=channels,
            embed_dim=embed_dim,
            periodic=periodic,
            transpose_decoder_when_nonperiodic=True,
        )


class ResnetBlock2D(ResnetBlockND):
    """2D ResNet block with additive time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int,
                 periodic: bool = True):
        super().__init__(in_ch, out_ch, embed_dim, spatial_dims=2,
                         periodic=periodic)


class AttnBlock2D(AttnBlockND):
    """Self-attention block for 2D data."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__(channels, spatial_dims=2, num_heads=num_heads)


class NCSNpp2D(NCSNppND):
    """NCSN++ style 2D U-Net with ResNet blocks and time conditioning."""

    def __init__(self, marginal_prob_std_fn, channels=[16, 32, 64, 128],
                 embed_dim=256, use_attention=False, periodic=True,
                 l_cond: bool = False):
        super().__init__(
            marginal_prob_std_fn,
            spatial_dims=2,
            channels=channels,
            embed_dim=embed_dim,
            use_attention=use_attention,
            periodic=periodic,
            l_cond=l_cond,
        )


__all__ = [
    "GaussianFourierProjection",
    "Dense",
    "ScoreNet",
    "ScoreNetUNet",
    "ScoreNetUNetPeriodic",
    "ResnetBlock2D",
    "AttnBlock2D",
    "NCSNpp2D",
]
