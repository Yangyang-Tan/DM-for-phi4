"""
Neural network architectures for 3D score-based diffusion models.

Classes:
    - ScoreNet3D: For 3D lattice field theory (no downsampling, supports periodic BC)
    - ScoreNet3DUNetPeriodic: 3D U-Net with downsampling and periodic BC
    - NCSNpp3D: Improved architecture based on Yang Song's "Improved Techniques 
                for Training Score-Based Generative Models" (NCSN++)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense3D(nn.Module):
    """A fully connected layer that reshapes outputs to 3D feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None, None]


class ScoreNet3DUNetPeriodic(nn.Module):
    """3D U-Net with downsampling/upsampling and periodic boundary conditions.
    
    Similar to 2D ScoreNetUNetPeriodic but uses 3D convolutions for 3D lattice fields.
    Uses strided convolutions for downsampling and nearest interpolation for upsampling.
    
    Args:
        marginal_prob_std_fn: Function that returns std at time t.
        channels: Channel sizes for each level [32, 64, 128, 256].
        embed_dim: Time embedding dimension.
    """

    def __init__(self, marginal_prob_std_fn, channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.act = lambda x: x * torch.sigmoid(x)  # swish activation

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoder (with downsampling via stride=2, circular padding for periodic BC)
        self.conv1 = nn.Conv3d(1, channels[0], 3, stride=1, padding=1, padding_mode="circular", bias=False)
        self.dense1 = Dense3D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(max(1, channels[0] // 8), num_channels=channels[0])

        self.conv2 = nn.Conv3d(channels[0], channels[1], 3, stride=2, padding=1, padding_mode="circular", bias=False)
        self.dense2 = Dense3D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(max(1, channels[1] // 8), num_channels=channels[1])

        self.conv3 = nn.Conv3d(channels[1], channels[2], 3, stride=2, padding=1, padding_mode="circular", bias=False)
        self.dense3 = Dense3D(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(max(1, channels[2] // 8), num_channels=channels[2])

        self.conv4 = nn.Conv3d(channels[2], channels[3], 3, stride=2, padding=1, padding_mode="circular", bias=False)
        self.dense4 = Dense3D(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(max(1, channels[3] // 8), num_channels=channels[3])

        # Decoder (with upsampling + skip connections)
        # h4 -> upsample -> concat h3 -> tconv4
        self.tconv4 = nn.Conv3d(channels[3] + channels[2], channels[2], 3, stride=1, padding=1, padding_mode="circular", bias=False)
        self.dense5 = Dense3D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(max(1, channels[2] // 8), num_channels=channels[2])

        # -> upsample -> concat h2 -> tconv3
        self.tconv3 = nn.Conv3d(channels[2] + channels[1], channels[1], 3, stride=1, padding=1, padding_mode="circular", bias=False)
        self.dense6 = Dense3D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(max(1, channels[1] // 8), num_channels=channels[1])

        # -> upsample -> concat h1 -> tconv2
        self.tconv2 = nn.Conv3d(channels[1] + channels[0], channels[0], 3, stride=1, padding=1, padding_mode="circular", bias=False)
        self.dense7 = Dense3D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(max(1, channels[0] // 8), num_channels=channels[0])

        # -> final output
        self.tconv1 = nn.Conv3d(channels[0], 1, 3, stride=1, padding=1, padding_mode="circular")

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoder with skip connections (upsample using nearest interpolation)
        # h4 -> upsample -> concat h3
        h = nn.functional.interpolate(h4, size=h3.shape[-3:], mode='nearest')
        h = self.tconv4(torch.cat([h, h3], dim=1))
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        # -> upsample -> concat h2
        h = nn.functional.interpolate(h, size=h2.shape[-3:], mode='nearest')
        h = self.tconv3(torch.cat([h, h2], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        # -> upsample -> concat h1
        h = nn.functional.interpolate(h, size=h1.shape[-3:], mode='nearest')
        h = self.tconv2(torch.cat([h, h1], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        # -> final output
        h = self.tconv1(h)

        # Note: 5 dimensions for 3D data: [batch, channel, D, H, W]
        return h / self.marginal_prob_std(t)[:, None, None, None, None]


class ScoreNet3D(nn.Module):
    """Time-dependent score-based model for 3D data built upon U-Net architecture.
    
    Uses 3D convolutions instead of 2D convolutions for processing 3D lattice fields.
    No downsampling version.
    
    Args:
        marginal_prob_std_fn: Function that returns std at time t.
        channels: Channel sizes for each level.
        embed_dim: Time embedding dimension.
        periodic: If True, use circular padding for periodic boundary conditions.
    """

    def __init__(
        self,
        marginal_prob_std_fn,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        periodic: bool = False,
    ):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.act = lambda x: x * torch.sigmoid(x)  # swish activation
        self.periodic = periodic

        def _conv3d(in_ch: int, out_ch: int, bias: bool = False) -> nn.Module:
            # Periodic boundary conditions are important for lattice systems.
            # Using circular padding avoids edge artifacts from zero-padding.
            if periodic:
                return nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="circular",
                    bias=bias,
                )
            return nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=bias)

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoder
        self.conv1 = _conv3d(1, channels[0], bias=False)
        self.dense1 = Dense3D(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = _conv3d(channels[0], channels[1], bias=False)
        self.dense2 = Dense3D(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = _conv3d(channels[1], channels[2], bias=False)
        self.dense3 = Dense3D(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = _conv3d(channels[2], channels[3], bias=False)
        self.dense4 = Dense3D(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoder
        # Note: stride=1 means no downsampling/upsampling; for periodic mode we use Conv3d
        # with circular padding to preserve translation invariance.
        self.tconv4 = _conv3d(channels[3], channels[2], bias=False)
        self.dense5 = Dense3D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = _conv3d(channels[2] * 2, channels[1], bias=False)
        self.dense6 = Dense3D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = _conv3d(channels[1] * 2, channels[0], bias=False)
        self.dense7 = Dense3D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = _conv3d(channels[0] * 2, 1, bias=True)

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder with skip connections
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], 1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], 1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], 1))

        # Note: 5 dimensions for 3D data: [batch, channel, D, H, W]
        return h / self.marginal_prob_std(t)[:, None, None, None, None]


# =============================================================================
# NCSN++ Style Architecture (Yang Song's Improved Techniques)
# =============================================================================

class ResnetBlock3D(nn.Module):
    """ResNet block with additive time conditioning.
    
    Note: skip_conv exists for channel projection but is NOT used for residual
    connection (residual was found to cause training instability).
    
    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        embed_dim: Time embedding dimension.
    """
    
    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode="circular", bias=False)
        self.norm1 = nn.GroupNorm(out_ch // 8, out_ch)
        
        # Time embedding projection (additive)
        self.temb_proj = nn.Linear(embed_dim, out_ch)
        
        # Second conv block
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, padding_mode="circular", bias=False)
        self.norm2 = nn.GroupNorm(out_ch // 8, out_ch)
        
        # Skip conv for channel mismatch (kept for checkpoint compatibility)
        if in_ch != out_ch:
            self.skip_conv = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # conv → add temb → norm → act
        h = self.conv1(x)
        h = h + self.temb_proj(temb)[:, :, None, None, None]
        h = self.act(self.norm1(h))
        
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        
        # No residual connection (removed for stability)
        return h


class AttnBlock3D(nn.Module):
    """Self-attention block for 3D data using PyTorch's MultiheadAttention.
    
    Args:
        channels: Number of input/output channels.
        num_heads: Number of attention heads (default 4).
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(channels // 8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # Initialize output projection to zero for residual connection
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        
        # Reshape: [B, C, D, H, W] -> [B, D*H*W, C]
        h = h.reshape(B, C, -1).permute(0, 2, 1)
        
        # Self-attention
        h, _ = self.attn(h, h, h, need_weights=False)
        
        # Reshape back: [B, D*H*W, C] -> [B, C, D, H, W]
        h = h.permute(0, 2, 1).reshape(B, C, D, H, W)
        
        return x + h


class NCSNpp3D(nn.Module):
    """NCSN++ style 3D U-Net with ResNet blocks and FiLM conditioning.
    
    Based on "Improved Techniques for Training Score-Based Generative Models" 
    by Yang Song et al. Key improvements: ResNet blocks, FiLM time conditioning,
    periodic BC, and zero initialization for residual paths.
    
    Args:
        marginal_prob_std_fn: Function that returns std at time t.
        channels: Channel sizes for each level [16, 32, 64, 128].
        embed_dim: Time embedding dimension.
        use_attention: Whether to use attention at bottleneck.
    """

    def __init__(self, marginal_prob_std_fn, channels=[10, 20, 40, 80], 
                 embed_dim=256, use_attention=False):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        
        # Time embedding (same as ScoreNet3DUNetPeriodic)
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoder with ResNet blocks
        self.conv_in = nn.Conv3d(1, channels[0], 3, padding=1, padding_mode="circular")
        self.res1a = ResnetBlock3D(channels[0], channels[0], embed_dim)
        self.res1b = ResnetBlock3D(channels[0], channels[0], embed_dim)
        self.down1 = nn.Conv3d(channels[0], channels[0], 3, stride=2, padding=1, padding_mode="circular")

        self.res2a = ResnetBlock3D(channels[0], channels[1], embed_dim)
        self.res2b = ResnetBlock3D(channels[1], channels[1], embed_dim)
        self.down2 = nn.Conv3d(channels[1], channels[1], 3, stride=2, padding=1, padding_mode="circular")

        self.res3a = ResnetBlock3D(channels[1], channels[2], embed_dim)
        self.res3b = ResnetBlock3D(channels[2], channels[2], embed_dim)
        self.down3 = nn.Conv3d(channels[2], channels[2], 3, stride=2, padding=1, padding_mode="circular")

        self.res4a = ResnetBlock3D(channels[2], channels[3], embed_dim)
        self.res4b = ResnetBlock3D(channels[3], channels[3], embed_dim)

        # Middle
        self.mid1 = ResnetBlock3D(channels[3], channels[3], embed_dim)
        self.mid_attn = AttnBlock3D(channels[3]) if use_attention else nn.Identity()
        self.mid2 = ResnetBlock3D(channels[3], channels[3], embed_dim)

        # Decoder with skip connections
        self.tres4a = ResnetBlock3D(channels[3] + channels[3], channels[3], embed_dim)
        self.tres4b = ResnetBlock3D(channels[3] + channels[3], channels[2], embed_dim)
        self.up3 = nn.Conv3d(channels[2], channels[2], 3, padding=1, padding_mode="circular")

        self.tres3a = ResnetBlock3D(channels[2] + channels[2], channels[2], embed_dim)
        self.tres3b = ResnetBlock3D(channels[2] + channels[2], channels[1], embed_dim)
        self.up2 = nn.Conv3d(channels[1], channels[1], 3, padding=1, padding_mode="circular")

        self.tres2a = ResnetBlock3D(channels[1] + channels[1], channels[1], embed_dim)
        self.tres2b = ResnetBlock3D(channels[1] + channels[1], channels[0], embed_dim)
        self.up1 = nn.Conv3d(channels[0], channels[0], 3, padding=1, padding_mode="circular")

        self.tres1a = ResnetBlock3D(channels[0] + channels[0], channels[0], embed_dim)
        self.tres1b = ResnetBlock3D(channels[0] + channels[0], channels[0], embed_dim)

        # Output
        self.norm_out = nn.GroupNorm(channels[0] // 8, channels[0])
        self.conv_out = nn.Conv3d(channels[0], 1, 3, padding=1, padding_mode="circular")

    def forward(self, x, t):
        temb = F.silu(self.embed(t))  # 激活，与 ScoreNet3DUNetPeriodic 一致

        # Encoder
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

        # Middle
        h = self.mid1(h4b, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        # Decoder with skip connections
        h = self.tres4a(torch.cat([h, h4b], dim=1), temb)
        h = self.tres4b(torch.cat([h, h4a], dim=1), temb)
        h = self.up3(F.interpolate(h, size=h3b.shape[-3:], mode='nearest'))

        h = self.tres3a(torch.cat([h, h3b], dim=1), temb)
        h = self.tres3b(torch.cat([h, h3a], dim=1), temb)
        h = self.up2(F.interpolate(h, size=h2b.shape[-3:], mode='nearest'))

        h = self.tres2a(torch.cat([h, h2b], dim=1), temb)
        h = self.tres2b(torch.cat([h, h2a], dim=1), temb)
        h = self.up1(F.interpolate(h, size=h1b.shape[-3:], mode='nearest'))

        h = self.tres1a(torch.cat([h, h1b], dim=1), temb)
        h = self.tres1b(torch.cat([h, h1a], dim=1), temb)

        # Output
        h = F.silu(self.norm_out(h))
        return self.conv_out(h) / self.marginal_prob_std(t)[:, None, None, None, None]


# Alias for backward compatibility
NCSNpp3DSimple = NCSNpp3D
