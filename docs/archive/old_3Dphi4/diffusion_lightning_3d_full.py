"""
3D Diffusion Model for Scalar Field Theory using PyTorch Lightning
Extension of DMasSQ for 3D lattice systems
"""

import functools
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# Set default dtype
torch.set_default_dtype(torch.float32)


# ==================== SDE Functions ====================

def marginal_prob_std(t, sigma):
    """Compute standard deviation of p_{0t}(x(t) | x(0))."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.float32)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute diffusion coefficient of the SDE."""
    if not isinstance(t, torch.Tensor):
        return torch.as_tensor(sigma**t, dtype=torch.float32)
    return sigma**t


# ==================== Network Components ====================

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
        # Output shape: (batch, output_dim, 1, 1, 1) for broadcasting with 3D feature maps
        return self.dense(x)[..., None, None, None]


class ScoreNet3D(nn.Module):
    """Time-dependent score-based model built upon 3D U-Net architecture."""

    def __init__(
        self,
        marginal_prob_std_fn,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        periodic: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std_fn
        self.act = lambda x: x * torch.sigmoid(x)  # swish activation
        self.periodic = periodic
        self.use_checkpoint = use_checkpoint  # Gradient checkpointing for memory savings

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
        # Note: stride=1 ConvTranspose3d doesn't upsample; for periodic mode we use Conv3d
        # with circular padding to preserve translation invariance.
        self.tconv4 = _conv3d(channels[3], channels[2], bias=False) if periodic else nn.ConvTranspose3d(channels[3], channels[2], 3, 1, 1, bias=False)
        self.dense5 = Dense3D(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = _conv3d(channels[2] * 2, channels[1], bias=False) if periodic else nn.ConvTranspose3d(channels[2] * 2, channels[1], 3, 1, 1, bias=False)
        self.dense6 = Dense3D(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = _conv3d(channels[1] * 2, channels[0], bias=False) if periodic else nn.ConvTranspose3d(channels[1] * 2, channels[0], 3, 1, 1, bias=False)
        self.dense7 = Dense3D(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = _conv3d(channels[0] * 2, 1, bias=True) if periodic else nn.ConvTranspose3d(channels[0] * 2, 1, 3, 1, 1)

    def _encoder_block1(self, x, embed):
        return self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))

    def _encoder_block2(self, h1, embed):
        return self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))

    def _encoder_block3(self, h2, embed):
        return self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))

    def _encoder_block4(self, h3, embed):
        return self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

    def _decoder_block4(self, h4, embed):
        return self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))

    def _decoder_block3(self, h, h3, embed):
        return self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], 1)) + self.dense6(embed)))

    def _decoder_block2(self, h, h2, embed):
        return self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], 1)) + self.dense7(embed)))

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory during training
            # This recomputes activations during backward pass instead of storing them
            h1 = checkpoint(self._encoder_block1, x, embed, use_reentrant=False)
            h2 = checkpoint(self._encoder_block2, h1, embed, use_reentrant=False)
            h3 = checkpoint(self._encoder_block3, h2, embed, use_reentrant=False)
            h4 = checkpoint(self._encoder_block4, h3, embed, use_reentrant=False)

            h = checkpoint(self._decoder_block4, h4, embed, use_reentrant=False)
            h = checkpoint(self._decoder_block3, h, h3, embed, use_reentrant=False)
            h = checkpoint(self._decoder_block2, h, h2, embed, use_reentrant=False)
            h = self.tconv1(torch.cat([h, h1], 1))
        else:
            # Standard forward pass (faster, more memory)
            h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
            h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
            h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
            h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

            h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
            h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], 1)) + self.dense6(embed)))
            h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], 1)) + self.dense7(embed)))
            h = self.tconv1(torch.cat([h, h1], 1))

        # Normalize by marginal std: shape (batch, 1, D, H, W)
        return h / self.marginal_prob_std(t)[:, None, None, None, None]


class Z2OddScore3D(nn.Module):
    """Enforce Z2 (phi -> -phi) oddness of the score: s(-x,t) = -s(x,t).

    If the target distribution is Z2-symmetric, the true score is odd. This wrapper
    enforces that symmetry exactly by antisymmetrizing an arbitrary base score model.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x2 = torch.cat([x, -x], dim=0)
        t2 = torch.cat([t, t], dim=0) if t.ndim else t.expand(2 * b)
        y2 = self.base(x2, t2)
        return 0.5 * (y2[:b] - y2[b:])


class CubicEquivariantScore3D(nn.Module):
    """Enforce cubic lattice (Oh) symmetry equivariance of the score.

    This wrapper symmetrizes by averaging over the octahedral group Oh (48 elements):
      s(x,t) = mean_g g^{-1} s_base(g x, t)
    which guarantees equivariance: s(gx,t) = g s(x,t).

    NOTE: This increases compute ~48x (group size), so use only if you need exact symmetry.
    For efficiency, we can use a subgroup (e.g., just rotations = 24 elements).
    """

    def __init__(self, base: nn.Module, full_group: bool = False):
        """
        Args:
            base: The base score network
            full_group: If True, use full Oh group (48 elements including reflections).
                       If False, use only rotations O (24 elements).
        """
        super().__init__()
        self.base = base
        self.full_group = full_group
        # Generate group elements
        self._generate_group_ops()

    def _generate_group_ops(self):
        """Generate the 24 rotation matrices for the cubic rotation group O.
        
        If full_group=True, also include reflections for Oh (48 elements).
        """
        ops = []
        
        # Identity and rotations about the coordinate axes
        # We represent each operation as a permutation of axes + signs
        # Format: (axis_perm, signs) where new_tensor[..., i] = signs[i] * old_tensor[..., axis_perm[i]]
        
        # Generate all 24 rotations of the cube
        # These are all combinations of:
        # - 6 face orientations (which axis points "up")
        # - 4 rotations around that axis
        
        # Axis permutations: all permutations of (0,1,2)
        perms = [
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
        ]
        
        # For each permutation, we need to pick signs that make it a proper rotation (det = +1)
        # A permutation (a,b,c) with signs (sa, sb, sc) has determinant = sign(perm) * sa * sb * sc
        # For proper rotation, we need det = +1
        
        for perm in perms:
            # Calculate permutation sign
            inv = 0
            for i in range(3):
                for j in range(i + 1, 3):
                    if perm[i] > perm[j]:
                        inv += 1
            perm_sign = 1 if inv % 2 == 0 else -1
            
            # Generate all sign combinations that give det = +1
            for s0 in [-1, 1]:
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        if perm_sign * s0 * s1 * s2 == 1:
                            ops.append((perm, (s0, s1, s2)))
        
        # Should have exactly 24 elements
        assert len(ops) == 24, f"Expected 24 rotations, got {len(ops)}"
        
        if self.full_group:
            # Add improper rotations (rotations * inversion)
            improper_ops = []
            for perm, signs in ops:
                improper_ops.append((perm, (-signs[0], -signs[1], -signs[2])))
            ops.extend(improper_ops)
            assert len(ops) == 48, f"Expected 48 elements for Oh, got {len(ops)}"
        
        self.ops = ops

    def _apply_g(self, x: torch.Tensor, op: tuple) -> torch.Tensor:
        """Apply a group element to a 3D tensor.
        
        x: shape (B, C, D, H, W)
        op: (axis_perm, signs)
        """
        perm, signs = op
        
        # First apply the permutation of spatial axes
        # dims (2,3,4) correspond to (D, H, W)
        y = x.permute(0, 1, perm[0] + 2, perm[1] + 2, perm[2] + 2)
        
        # Then apply sign flips (reflections) along each axis
        if signs[0] == -1:
            y = torch.flip(y, dims=(2,))
        if signs[1] == -1:
            y = torch.flip(y, dims=(3,))
        if signs[2] == -1:
            y = torch.flip(y, dims=(4,))
        
        return y

    def _apply_g_inv(self, y: torch.Tensor, op: tuple) -> torch.Tensor:
        """Apply the inverse of a group element.
        
        For orthogonal transformations, g^{-1} = g^T.
        We need to undo: flip -> permute, so inverse is: inverse_permute -> flip
        """
        perm, signs = op
        
        # First undo the flips (flips are self-inverse)
        if signs[2] == -1:
            y = torch.flip(y, dims=(4,))
        if signs[1] == -1:
            y = torch.flip(y, dims=(3,))
        if signs[0] == -1:
            y = torch.flip(y, dims=(2,))
        
        # Then undo the permutation
        inv_perm = [0, 0, 0]
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        y = y.permute(0, 1, inv_perm[0] + 2, inv_perm[1] + 2, inv_perm[2] + 2)
        
        return y

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        n_ops = len(self.ops)
        
        xs: list[torch.Tensor] = []
        for op in self.ops:
            xs.append(self._apply_g(x, op))

        x_cat = torch.cat(xs, dim=0)  # (n_ops * B, C, D, H, W)
        t_cat = torch.cat([t] * n_ops, dim=0) if t.ndim else t.expand(n_ops * b)
        y_cat = self.base(x_cat, t_cat)

        ys = torch.chunk(y_cat, n_ops, dim=0)
        y_can = [self._apply_g_inv(y, op) for y, op in zip(ys, self.ops)]
        return torch.stack(y_can, dim=0).mean(dim=0)


# ==================== Lightning Module ====================

class DiffusionModel3D(pl.LightningModule):
    """PyTorch Lightning module for 3D score-based diffusion model."""

    def __init__(
        self,
        sigma=25.0,
        lr=1e-3,
        L=16,
        periodic: bool = False,
        symmetry_aug: bool = False,
        z2_arch: bool = False,
        cubic_arch: bool = False,
        full_cubic_group: bool = False,
        # Memory optimization options
        use_checkpoint: bool = False,
        accumulate_grad_batches: int = 1,
    ):
        """
        Args:
            sigma: SDE noise scale
            lr: Learning rate
            L: Lattice size (L x L x L)
            periodic: Use periodic boundary conditions in convolutions
            symmetry_aug: Apply random symmetry augmentations during training
            z2_arch: Use Z2-symmetric score architecture (phi -> -phi)
            cubic_arch: Use cubic-equivariant score architecture
            full_cubic_group: If cubic_arch, use full Oh group (48) vs O group (24)
            
            # Memory optimization options:
            use_checkpoint: Use gradient checkpointing to reduce memory (~30-50% savings)
                           Trades compute for memory by recomputing activations in backward pass.
            accumulate_grad_batches: Accumulate gradients over N batches before optimizer step.
                                    Effectively increases batch size without memory increase.
                                    Set this in pl.Trainer instead for proper integration.
        """
        super().__init__()
        self.save_hyperparameters()

        self.sigma = sigma
        self.lr = lr
        self.L = L
        self.symmetry_aug = symmetry_aug
        self.z2_arch = z2_arch
        self.cubic_arch = cubic_arch
        self.use_checkpoint = use_checkpoint

        # Create SDE functions
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

        # Create score model with checkpointing option
        score = ScoreNet3D(self.marginal_prob_std_fn, periodic=periodic, use_checkpoint=use_checkpoint)
        if cubic_arch:
            score = CubicEquivariantScore3D(score, full_group=full_cubic_group)
        if z2_arch:
            score = Z2OddScore3D(score)
        self.score_model = score

    def _apply_symmetry_aug(self, x: torch.Tensor, z: torch.Tensor):
        """Apply simple physical symmetries (3D lattice).

        This enforces:
          - translation invariance (random periodic shifts)
          - lattice rotations/reflections (cubic group)
          - Z2 symmetry (phi -> -phi) via per-sample sign flips
        """
        # IMPORTANT: sample augmentation params on CPU to avoid GPU sync stalls.
        # Random periodic translation (same shift for whole batch; fast, still enforces invariance)
        dx = int(torch.randint(0, self.L, (1,), device="cpu").item())
        dy = int(torch.randint(0, self.L, (1,), device="cpu").item())
        dz = int(torch.randint(0, self.L, (1,), device="cpu").item())
        x = torch.roll(x, shifts=(dx, dy, dz), dims=(2, 3, 4))
        z = torch.roll(z, shifts=(dx, dy, dz), dims=(2, 3, 4))

        # Random 90-degree rotation around a random axis
        # Choose a random axis (0, 1, 2) and rotation count (0, 1, 2, 3)
        axis = int(torch.randint(0, 3, (1,), device="cpu").item())
        k = int(torch.randint(0, 4, (1,), device="cpu").item())
        if k:
            # Rotate around the chosen axis
            if axis == 0:
                dims = (3, 4)  # rotate in YZ plane
            elif axis == 1:
                dims = (2, 4)  # rotate in XZ plane
            else:
                dims = (2, 3)  # rotate in XY plane
            x = torch.rot90(x, k, dims=dims)
            z = torch.rot90(z, k, dims=dims)

        # Random reflections along each axis
        if float(torch.rand((), device="cpu").item()) < 0.5:
            x = torch.flip(x, dims=(2,))
            z = torch.flip(z, dims=(2,))
        if float(torch.rand((), device="cpu").item()) < 0.5:
            x = torch.flip(x, dims=(3,))
            z = torch.flip(z, dims=(3,))
        if float(torch.rand((), device="cpu").item()) < 0.5:
            x = torch.flip(x, dims=(4,))
            z = torch.flip(z, dims=(4,))

        return x, z

    def _apply_symmetry_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same physical symmetries to generated samples.

        This is a post-processing step to remove residual symmetry-breaking in finite samples
        (useful for per-site cumulant maps and to stabilize odd moments).
        """
        # IMPORTANT: sample transform params on CPU to avoid GPU sync stalls.
        dx = int(torch.randint(0, self.L, (1,), device="cpu").item())
        dy = int(torch.randint(0, self.L, (1,), device="cpu").item())
        dz = int(torch.randint(0, self.L, (1,), device="cpu").item())
        x = torch.roll(x, shifts=(dx, dy, dz), dims=(2, 3, 4))

        # Random rotation
        axis = int(torch.randint(0, 3, (1,), device="cpu").item())
        k = int(torch.randint(0, 4, (1,), device="cpu").item())
        if k:
            if axis == 0:
                dims = (3, 4)
            elif axis == 1:
                dims = (2, 4)
            else:
                dims = (2, 3)
            x = torch.rot90(x, k, dims=dims)

        # Random reflections
        if float(torch.rand((), device="cpu").item()) < 0.5:
            x = torch.flip(x, dims=(2,))
        if float(torch.rand((), device="cpu").item()) < 0.5:
            x = torch.flip(x, dims=(3,))
        if float(torch.rand((), device="cpu").item()) < 0.5:
            x = torch.flip(x, dims=(4,))

        # Per-sample Z2 sign flip (phi -> -phi)
        signs = (torch.randint(0, 2, (x.shape[0], 1, 1, 1, 1), device=x.device, dtype=x.dtype) * 2 - 1)
        return x * signs

    def forward(self, x, t):
        return self.score_model(x, t)

    def loss_fn(self, x, eps=1e-5):
        """Compute denoising score matching loss."""
        # Sample a different time `t` for each training example in the batch.
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)
        std = self.marginal_prob_std_fn(random_t)
        if self.symmetry_aug:
            x, z = self._apply_symmetry_aug(x, z)
        perturbed_x = x + z * std[:, None, None, None, None]
        score = self(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1, 2, 3, 4)))
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.loss_fn(x)
        # Log step loss for progress bar, epoch average for checkpointing
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(
        self,
        num_samples=64,
        num_steps=200,
        eps=1e-3,
        sample_batch_size=None,
        show_progress=True,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ):
        """Generate samples from the diffusion model.
        
        Args:
            num_samples: Total number of samples to generate
            num_steps: Number of diffusion steps
            eps: Small value to avoid numerical issues at t=0
            sample_batch_size: Batch size for sampling (None = all at once)
            show_progress: Show progress bar
            use_amp: Use automatic mixed precision for faster/lower-memory sampling
            amp_dtype: Dtype for AMP (torch.float16 or torch.bfloat16)
        """
        self.eval()

        device = self.device
        if sample_batch_size is None:
            sample_batch_size = num_samples

        # Precompute common scalars/tensors on device
        time_steps = torch.linspace(1.0, eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        sqrt_step_size = torch.sqrt(step_size)
        init_std = self.marginal_prob_std_fn(torch.ones((), device=device))

        def _sample_batch(batch_n: int, desc: str | None = None, leave: bool = True) -> torch.Tensor:
            # 3D: shape is (batch, 1, L, L, L)
            x = torch.randn(batch_n, 1, self.L, self.L, self.L, device=device) * init_std
            noise = torch.empty_like(x)

            step_iter = time_steps
            if show_progress:
                step_iter = tqdm(step_iter, desc=desc or "Sampling", leave=leave)

            # inference_mode reduces overhead vs no_grad()
            with torch.inference_mode():
                for i, time_step in enumerate(step_iter):
                    # Avoid allocating a fresh t-vector each step (expand is a view).
                    batch_time_step = time_step.expand(batch_n)

                    # g is a scalar at this time step; broadcast in arithmetic.
                    g = self.diffusion_coeff_fn(time_step)
                    if i != num_steps-1:
                        noise.normal_()
                        noise.mul_(sqrt_step_size * g)
                        x.add_(noise)
                    
                    # Use AMP for score computation if enabled
                    if use_amp:
                        with torch.autocast(device_type='cuda', dtype=amp_dtype):
                            score = self(x, batch_time_step)
                        score = score.float()  # Convert back to float32 for accumulation
                    else:
                        score = self(x, batch_time_step)

                    # Drift update: x <- x + g(t)^2 * score(x,t) * dt
                    score.mul_((g * g) * step_size)
                    x.add_(score)

                    # Diffusion update: x <- x + sqrt(dt) * g(t) * N(0, I)
                    # Skip the final noise term (matches returning mean_x in the original code).

            if self.symmetry_aug:
                x = self._apply_symmetry_sample(x)
            return x

        if sample_batch_size >= num_samples:
            return _sample_batch(num_samples, desc="Sampling", leave=True)

        out = torch.empty(num_samples, 1, self.L, self.L, self.L, device=device)
        for start in range(0, num_samples, sample_batch_size):
            batch_n = min(sample_batch_size, num_samples - start)
            desc = f"Sampling [{start + batch_n}/{num_samples}]"
            out[start : start + batch_n] = _sample_batch(batch_n, desc=desc, leave=False)

        return out

    @torch.no_grad()
    def sample_mala(
        self,
        num_samples=64,
        num_steps=200,
        eps=1e-3,
        sample_batch_size=None,
        show_progress=True,
        k=0.5,
        l=2.5,
        mh_last_steps=10,
        alpha=0.01,
    ):
        """Sample using MAALA: 中间用 EM，最后几步用 MALA+MH 校正 (3D版本)。
        
        Args:
            num_samples: 样本数量
            num_steps: 扩散步数
            eps: 最小时间步
            sample_batch_size: 每批生成的样本数 (None = 一次性全部生成)
            show_progress: 是否显示进度条
            k, l: φ⁴ 理论参数
            mh_last_steps: 最后多少步启用 MH 校正
            alpha: MH 步长缩放因子
            
        Returns:
            samples: 生成的样本
            acceptance_rate: 平均 MH 接受率
        """
        self.eval()
        device = self.device
        
        if sample_batch_size is None:
            sample_batch_size = num_samples
        
        # 3D φ⁴ action
        def action(phi):
            neighbors = (torch.roll(phi, 1, 2) + torch.roll(phi, -1, 2) +
                        torch.roll(phi, 1, 3) + torch.roll(phi, -1, 3) +
                        torch.roll(phi, 1, 4) + torch.roll(phi, -1, 4))
            phi_sq = phi ** 2
            return torch.sum(-k * phi * neighbors + (1 - 2*l) * phi_sq + l * phi_sq**2, dim=(1,2,3,4))
        
        time_steps = torch.linspace(1.0, eps, num_steps, device=device)
        dt = time_steps[0] - time_steps[1]
        init_std = self.marginal_prob_std_fn(torch.ones((), device=device))
        
        def _sample_batch(batch_n, desc=None, leave=True):
            def mh_step(x, y, noise, t, h):
                batch_t = t.expand(batch_n)
                score_x, score_y = self(x, batch_t), self(y, batch_t)
                drift = h * (score_x + score_y)
                sqrt_2h = torch.sqrt(2 * h)
                log_q = -0.25/h * torch.sum((sqrt_2h * noise + drift)**2, dim=(1,2,3,4)) + \
                        0.5 * torch.sum(noise**2, dim=(1,2,3,4))
                log_pi = action(x) - action(y)
                accept_prob = torch.clamp(torch.exp(log_q + log_pi), 0, 1)
                accept = torch.rand(batch_n, device=device) < accept_prob
                x_new = torch.where(accept.view(-1,1,1,1,1), y, x)
                return x_new, accept.float().mean().item()
            
            x = torch.randn(batch_n, 1, self.L, self.L, self.L, device=device) * init_std
            total_acc, mh_count = 0.0, 0
            
            step_iter = enumerate(time_steps)
            if show_progress:
                step_iter = tqdm(step_iter, total=num_steps, desc=desc or "MAALA", leave=leave)
            
            with torch.inference_mode():
                for i, t in step_iter:
                    batch_t = t.expand(batch_n)
                    in_mh_phase = (num_steps - 1 - i) < mh_last_steps
                    
                    g = self.diffusion_coeff_fn(t)
                    
                    if not in_mh_phase:
                        if i < num_steps - 1:
                            x = x + torch.randn_like(x) * torch.sqrt(dt) * g
                        x = x + self(x, batch_t) * (g**2 * dt)
                    else:
                        # MH: alpha * t² or alpha * g² * dt
                        h = alpha * (g ** 2) * dt
                        score = self(x, batch_t)
                        noise = torch.randn_like(x)
                        y = x + h * score + torch.sqrt(2 * h) * noise
                        x, acc = mh_step(x, y, noise, t, h)
                        total_acc += acc
                        mh_count += 1
            
            if self.symmetry_aug:
                x = self._apply_symmetry_sample(x)
            
            return x, total_acc / max(mh_count, 1)
        
        # 单批处理
        if sample_batch_size >= num_samples:
            samples, avg_acc = _sample_batch(num_samples, desc="MAALA", leave=True)
            if show_progress:
                print(f"MH acceptance rate: {avg_acc:.4f}")
            return samples, avg_acc
        
        # 分批处理
        out = torch.empty(num_samples, 1, self.L, self.L, self.L, device=device)
        total_acc_all = 0.0
        n_batches = 0
        
        for start in range(0, num_samples, sample_batch_size):
            batch_n = min(sample_batch_size, num_samples - start)
            desc = f"MAALA [{start + batch_n}/{num_samples}]"
            batch_samples, batch_acc = _sample_batch(batch_n, desc=desc, leave=False)
            out[start : start + batch_n] = batch_samples
            total_acc_all += batch_acc
            n_batches += 1
        
        avg_acc = total_acc_all / n_batches
        if show_progress:
            print(f"MH acceptance rate: {avg_acc:.4f}")
        
        return out, avg_acc


# ==================== Data Module ====================

class FieldDataModule3D(pl.LightningDataModule):
    """PyTorch Lightning DataModule for 3D field configurations."""

    def __init__(self, data_path, batch_size=64, normalize=True):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.normalize = normalize

        self.cfgs_min = None
        self.cfgs_max = None

    def setup(self, stage=None):
        import h5py
        with h5py.File(self.data_path, "r") as f:
            cfgs = np.array(f["cfgs"])

        # Normalize if needed
        if self.normalize:
            self.cfgs_min = cfgs.min()
            self.cfgs_max = cfgs.max()
            cfgs = ((cfgs - self.cfgs_min) / (self.cfgs_max - self.cfgs_min) - 0.5) * 2

        # Convert to tensors: expect shape (N, D, H, W) -> (N, 1, D, H, W)
        configs = torch.from_numpy(cfgs).unsqueeze(1).float()

        # Training data (no separate test/validation split here)
        self.train_data = TensorDataset(configs, configs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def renorm(self, y):
        """Renormalize data back to original scale."""
        if self.cfgs_min is None:
            return y
        return (y / 2 + 0.5) * (self.cfgs_max - self.cfgs_min) + self.cfgs_min


# ==================== Utility Functions ====================

def grab(var):
    """Convert tensor to numpy array."""
    return var.detach().cpu().numpy()


def jackknife(samples: np.ndarray):
    """Return mean and estimated error using jackknife."""
    means = []
    for i in range(samples.shape[0]):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))
    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))
    return mean, error


def get_mag_3d(cfgs: np.ndarray):
    """Return mean and error of magnetization for 3D configs."""
    # cfgs shape: (N, D, H, W) or (N, 1, D, H, W)
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    return jackknife(cfgs.mean(axis=axis))


def get_abs_mag_3d(cfgs: np.ndarray):
    """Return mean and error of absolute magnetization for 3D configs."""
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    return jackknife(np.abs(cfgs.mean(axis=axis)))


def get_chi2_3d(cfgs: np.ndarray):
    """Return mean and error of susceptibility for 3D configs."""
    V = np.prod(cfgs.shape[1:])
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    mags = cfgs.mean(axis=axis)
    return jackknife(V * (mags**2 - mags.mean()**2))


def get_UL_3d(cfgs: np.ndarray):
    """Return mean and error of Binder cumulant for 3D configs."""
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    mags = cfgs.mean(axis=axis)
    return jackknife(1 - mags**4 / 3 / (mags**2).mean()**2)


def mag_3d(cfgs: np.ndarray):
    """Return magnetization per configuration for 3D configs."""
    return cfgs.mean(axis=(1, 2, 3))

