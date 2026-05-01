"""
Dimension-agnostic neural network blocks for score-based diffusion models.

The public 2D and 3D modules keep their historical class names in
``networks.py`` and ``3Dphi4/networks_3d.py``.  This file contains the shared
implementations so the two paths do not drift.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_nd(spatial_dims: int):
    return nn.Conv2d if spatial_dims == 2 else nn.Conv3d


def _conv_transpose_nd(spatial_dims: int):
    return nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d


def _spatial_slice(spatial_dims: int):
    return slice(-spatial_dims, None)


def _std_view(t: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    return t.view((-1,) + (1,) * (spatial_dims + 1))


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DenseND(nn.Module):
    """Fully connected layer reshaped to 2D/3D feature maps."""

    def __init__(self, input_dim: int, output_dim: int, spatial_dims: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.spatial_dims = spatial_dims

    def forward(self, x):
        return self.dense(x)[(...,) + (None,) * self.spatial_dims]


class ScoreNetPeriodicUNetND(nn.Module):
    """Periodic U-Net with strided downsampling and nearest upsampling."""

    def __init__(self, marginal_prob_std_fn, spatial_dims: int,
                 channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.spatial_dims = spatial_dims
        self.act = lambda x: x * torch.sigmoid(x)
        conv = _conv_nd(spatial_dims)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = conv(1, channels[0], 3, stride=1, padding=1,
                          padding_mode="circular", bias=False)
        self.dense1 = DenseND(embed_dim, channels[0], spatial_dims)
        self.gnorm1 = nn.GroupNorm(max(1, channels[0] // 8), channels[0])

        self.conv2 = conv(channels[0], channels[1], 3, stride=2, padding=1,
                          padding_mode="circular", bias=False)
        self.dense2 = DenseND(embed_dim, channels[1], spatial_dims)
        self.gnorm2 = nn.GroupNorm(max(1, channels[1] // 8), channels[1])

        self.conv3 = conv(channels[1], channels[2], 3, stride=2, padding=1,
                          padding_mode="circular", bias=False)
        self.dense3 = DenseND(embed_dim, channels[2], spatial_dims)
        self.gnorm3 = nn.GroupNorm(max(1, channels[2] // 8), channels[2])

        self.conv4 = conv(channels[2], channels[3], 3, stride=2, padding=1,
                          padding_mode="circular", bias=False)
        self.dense4 = DenseND(embed_dim, channels[3], spatial_dims)
        self.gnorm4 = nn.GroupNorm(max(1, channels[3] // 8), channels[3])

        self.tconv4 = conv(channels[3] + channels[2], channels[2], 3,
                           stride=1, padding=1, padding_mode="circular",
                           bias=False)
        self.dense5 = DenseND(embed_dim, channels[2], spatial_dims)
        self.tgnorm4 = nn.GroupNorm(max(1, channels[2] // 8), channels[2])

        self.tconv3 = conv(channels[2] + channels[1], channels[1], 3,
                           stride=1, padding=1, padding_mode="circular",
                           bias=False)
        self.dense6 = DenseND(embed_dim, channels[1], spatial_dims)
        self.tgnorm3 = nn.GroupNorm(max(1, channels[1] // 8), channels[1])

        self.tconv2 = conv(channels[1] + channels[0], channels[0], 3,
                           stride=1, padding=1, padding_mode="circular",
                           bias=False)
        self.dense7 = DenseND(embed_dim, channels[0], spatial_dims)
        self.tgnorm2 = nn.GroupNorm(max(1, channels[0] // 8), channels[0])

        self.tconv1 = conv(channels[0], 1, 3, stride=1, padding=1,
                           padding_mode="circular")

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        h = F.interpolate(h4, size=h3.shape[_spatial_slice(self.spatial_dims)],
                          mode="nearest")
        h = self.act(self.tgnorm4(self.tconv4(torch.cat([h, h3], dim=1))
                                  + self.dense5(embed)))

        h = F.interpolate(h, size=h2.shape[_spatial_slice(self.spatial_dims)],
                          mode="nearest")
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h2], dim=1))
                                  + self.dense6(embed)))

        h = F.interpolate(h, size=h1.shape[_spatial_slice(self.spatial_dims)],
                          mode="nearest")
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h1], dim=1))
                                  + self.dense7(embed)))

        h = self.tconv1(h)
        return h / _std_view(self.marginal_prob_std(t), self.spatial_dims)


class ScoreNetND(nn.Module):
    """No-downsampling score model shared by 2D and 3D phi4."""

    def __init__(self, marginal_prob_std_fn, spatial_dims: int,
                 channels=[32, 64, 128, 256], embed_dim=256,
                 periodic: bool = False,
                 transpose_decoder_when_nonperiodic: bool = False):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.spatial_dims = spatial_dims
        self.act = lambda x: x * torch.sigmoid(x)
        self.periodic = periodic
        conv = _conv_nd(spatial_dims)
        tconv = _conv_transpose_nd(spatial_dims)

        def make_conv(in_ch: int, out_ch: int, bias: bool = False) -> nn.Module:
            if periodic:
                return conv(in_ch, out_ch, 3, 1, 1, padding_mode="circular",
                            bias=bias)
            return conv(in_ch, out_ch, 3, 1, 1, bias=bias)

        def make_decoder(in_ch: int, out_ch: int, bias: bool = False) -> nn.Module:
            if periodic or not transpose_decoder_when_nonperiodic:
                return make_conv(in_ch, out_ch, bias=bias)
            return tconv(in_ch, out_ch, 3, 1, 1, bias=bias)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = make_conv(1, channels[0], bias=False)
        self.dense1 = DenseND(embed_dim, channels[0], spatial_dims)
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = make_conv(channels[0], channels[1], bias=False)
        self.dense2 = DenseND(embed_dim, channels[1], spatial_dims)
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = make_conv(channels[1], channels[2], bias=False)
        self.dense3 = DenseND(embed_dim, channels[2], spatial_dims)
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = make_conv(channels[2], channels[3], bias=False)
        self.dense4 = DenseND(embed_dim, channels[3], spatial_dims)
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        self.tconv4 = make_decoder(channels[3], channels[2], bias=False)
        self.dense5 = DenseND(embed_dim, channels[2], spatial_dims)
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = make_decoder(channels[2] * 2, channels[1], bias=False)
        self.dense6 = DenseND(embed_dim, channels[1], spatial_dims)
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = make_decoder(channels[1] * 2, channels[0], bias=False)
        self.dense7 = DenseND(embed_dim, channels[0], spatial_dims)
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = make_decoder(channels[0] * 2, 1, bias=True)

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], 1))
                                  + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], 1))
                                  + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], 1))

        return h / _std_view(self.marginal_prob_std(t), self.spatial_dims)


class ResnetBlockND(nn.Module):
    """ResNet block with additive time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int,
                 spatial_dims: int, periodic: bool = True):
        super().__init__()
        self.spatial_dims = spatial_dims
        conv = _conv_nd(spatial_dims)
        pad_mode = "circular" if periodic else "zeros"

        self.conv1 = conv(in_ch, out_ch, 3, padding=1,
                          padding_mode=pad_mode, bias=False)
        self.norm1 = nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.temb_proj = nn.Linear(embed_dim, out_ch)

        self.conv2 = conv(out_ch, out_ch, 3, padding=1,
                          padding_mode=pad_mode, bias=False)
        self.norm2 = nn.GroupNorm(max(1, out_ch // 8), out_ch)

        if in_ch != out_ch:
            self.skip_conv = conv(in_ch, out_ch, 1, bias=False)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + temb_proj_view(self.temb_proj(temb), self.spatial_dims)
        h = self.act(self.norm1(h))
        h = self.act(self.norm2(self.conv2(h)))
        return h


def temb_proj_view(temb: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    return temb[(...,) + (None,) * spatial_dims]


class AttnBlockND(nn.Module):
    """Self-attention block over all lattice/image sites."""

    def __init__(self, channels: int, spatial_dims: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.spatial_dims = spatial_dims
        self.norm = nn.GroupNorm(max(1, channels // 8), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        h = self.norm(x)
        h = h.reshape(b, c, -1).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(b, c, *spatial_shape)
        return x + h


class NCSNppND(nn.Module):
    """NCSN++ style U-Net with ResNet blocks and time conditioning."""

    def __init__(self, marginal_prob_std_fn, spatial_dims: int,
                 channels=[16, 32, 64, 128], embed_dim=256,
                 use_attention=False, periodic=True, l_cond: bool = False):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.spatial_dims = spatial_dims
        self.l_cond = l_cond
        conv = _conv_nd(spatial_dims)
        pad_mode = "circular" if periodic else "zeros"

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Optional lattice-size conditioning. Adds a Gaussian-Fourier embedding
        # of 1/L (read off the input shape at forward time) to the time
        # embedding. Lets a single set of weights specialise per L when
        # trained on multiple lattice sizes.
        if l_cond:
            self.L_embed = nn.Sequential(
                GaussianFourierProjection(embed_dim=embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )

        self.conv_in = conv(1, channels[0], 3, padding=1,
                            padding_mode=pad_mode)
        self.res1a = ResnetBlockND(channels[0], channels[0], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.res1b = ResnetBlockND(channels[0], channels[0], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.down1 = conv(channels[0], channels[0], 3, stride=2, padding=1,
                          padding_mode=pad_mode)

        self.res2a = ResnetBlockND(channels[0], channels[1], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.res2b = ResnetBlockND(channels[1], channels[1], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.down2 = conv(channels[1], channels[1], 3, stride=2, padding=1,
                          padding_mode=pad_mode)

        self.res3a = ResnetBlockND(channels[1], channels[2], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.res3b = ResnetBlockND(channels[2], channels[2], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.down3 = conv(channels[2], channels[2], 3, stride=2, padding=1,
                          padding_mode=pad_mode)

        self.res4a = ResnetBlockND(channels[2], channels[3], embed_dim,
                                   spatial_dims, periodic=periodic)
        self.res4b = ResnetBlockND(channels[3], channels[3], embed_dim,
                                   spatial_dims, periodic=periodic)

        self.mid1 = ResnetBlockND(channels[3], channels[3], embed_dim,
                                  spatial_dims, periodic=periodic)
        self.mid_attn = (AttnBlockND(channels[3], spatial_dims)
                         if use_attention else nn.Identity())
        self.mid2 = ResnetBlockND(channels[3], channels[3], embed_dim,
                                  spatial_dims, periodic=periodic)

        self.tres4a = ResnetBlockND(channels[3] + channels[3], channels[3],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.tres4b = ResnetBlockND(channels[3] + channels[3], channels[2],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.up3 = conv(channels[2], channels[2], 3, padding=1,
                        padding_mode=pad_mode)

        self.tres3a = ResnetBlockND(channels[2] + channels[2], channels[2],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.tres3b = ResnetBlockND(channels[2] + channels[2], channels[1],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.up2 = conv(channels[1], channels[1], 3, padding=1,
                        padding_mode=pad_mode)

        self.tres2a = ResnetBlockND(channels[1] + channels[1], channels[1],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.tres2b = ResnetBlockND(channels[1] + channels[1], channels[0],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.up1 = conv(channels[0], channels[0], 3, padding=1,
                        padding_mode=pad_mode)

        self.tres1a = ResnetBlockND(channels[0] + channels[0], channels[0],
                                    embed_dim, spatial_dims, periodic=periodic)
        self.tres1b = ResnetBlockND(channels[0] + channels[0], channels[0],
                                    embed_dim, spatial_dims, periodic=periodic)

        self.norm_out = nn.GroupNorm(max(1, channels[0] // 8), channels[0])
        self.conv_out = conv(channels[0], 1, 3, padding=1,
                             padding_mode=pad_mode)

    def forward(self, x, t):
        if self.l_cond:
            L_inv = torch.full_like(t, 1.0 / x.shape[-1])
            temb = F.silu(self.embed(t) + self.L_embed(L_inv))
        else:
            temb = F.silu(self.embed(t))

        h = self.conv_in(x)
        h1a = self.res1a(h, temb)
        h1b = self.res1b(h1a, temb)
        h = self.down1(h1b)

        h2a = self.res2a(h, temb)
        h2b = self.res2b(h2a, temb)
        h = self.down2(h2b)

        h3a = self.res3a(h, temb)
        h3b = self.res3b(h3a, temb)
        h = self.down3(h3b)

        h4a = self.res4a(h, temb)
        h4b = self.res4b(h4a, temb)

        h = self.mid1(h4b, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        h = self.tres4a(torch.cat([h, h4b], dim=1), temb)
        h = self.tres4b(torch.cat([h, h4a], dim=1), temb)
        h = self.up3(F.interpolate(
            h, size=h3b.shape[_spatial_slice(self.spatial_dims)], mode="nearest"
        ))

        h = self.tres3a(torch.cat([h, h3b], dim=1), temb)
        h = self.tres3b(torch.cat([h, h3a], dim=1), temb)
        h = self.up2(F.interpolate(
            h, size=h2b.shape[_spatial_slice(self.spatial_dims)], mode="nearest"
        ))

        h = self.tres2a(torch.cat([h, h2b], dim=1), temb)
        h = self.tres2b(torch.cat([h, h2a], dim=1), temb)
        h = self.up1(F.interpolate(
            h, size=h1b.shape[_spatial_slice(self.spatial_dims)], mode="nearest"
        ))

        h = self.tres1a(torch.cat([h, h1b], dim=1), temb)
        h = self.tres1b(torch.cat([h, h1a], dim=1), temb)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h) / _std_view(self.marginal_prob_std(t),
                                            self.spatial_dims)
