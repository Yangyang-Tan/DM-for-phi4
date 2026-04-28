"""
PyTorch Lightning diffusion model for 3D data with EMA support.
"""

import functools

import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm


def marginal_prob_std(t, sigma):
    """Compute standard deviation of p_{0t}(x(t) | x(0))."""
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute diffusion coefficient of the SDE."""
    return sigma**t


class DiffusionModel3D(pl.LightningModule):
    """Score-based diffusion model for 3D data with EMA.
    
    Args:
        score_model: Neural network that predicts the score function.
        sigma: Noise scale for the diffusion process.
        lr: Learning rate.
        L: Lattice size (for 3D, generates L x L x L samples).
        ema_decay: Exponential moving average decay rate.
        ema_start_epoch: Epoch to start updating EMA.
        norm_min: Minimum value for normalization.
        norm_max: Maximum value for normalization.
    """

    def __init__(self, score_model, sigma=25.0, lr=1e-3, L=32, ema_decay=0.999,
                 ema_start_epoch=0, norm_min=None, norm_max=None):
        super().__init__()
        self.save_hyperparameters(ignore=['score_model'])
        self.sigma = sigma
        self.lr = lr
        self.L = L
        self.ema_start_epoch = ema_start_epoch
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.score_model = score_model
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=ema_decay)

    def forward(self, x, t):
        return self.score_model(x, t)

    def loss_fn(self, x, eps=1e-5):
        """Compute the score matching loss for 3D data."""
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)
        std = self.marginal_prob_std_fn(random_t)
        perturbed_x = x + z * std[:, None, None, None, None]
        score = self(perturbed_x, random_t)
        per_sample_loss = torch.sum((score * std[:, None, None, None, None] + z)**2, dim=(1, 2, 3, 4))
        return torch.mean(per_sample_loss), per_sample_loss.detach(), random_t.detach()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, per_sample_loss, random_t = self.loss_fn(x)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, prog_bar=True, on_step=False, on_epoch=True)

        t_bins = [
            (0.0, 0.2, 'loss_UV'),
            (0.2, 0.8, 'loss_mid'),
            (0.8, 1.01, 'loss_IR'),
        ]
        for t_lo, t_hi, name in t_bins:
            mask = (random_t >= t_lo) & (random_t < t_hi)
            if mask.any():
                self.log(name, per_sample_loss[mask].mean(),
                         on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # EMA hooks
    def on_fit_start(self):
        self.ema.to(self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch >= self.ema_start_epoch:
            self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Handle torch.compile's _orig_mod. prefix mismatch:
        # adapt checkpoint keys to match current model (compiled or not)
        sd = checkpoint.get('state_dict', {})
        model_keys = set(self.state_dict().keys())
        ckpt_has_orig = any('._orig_mod.' in k for k in sd)
        model_has_orig = any('._orig_mod.' in k for k in model_keys)

        if ckpt_has_orig and not model_has_orig:
            sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
        elif not ckpt_has_orig and model_has_orig:
            sd = {k.replace('score_model.', 'score_model._orig_mod.', 1): v
                  for k, v in sd.items()}
        checkpoint['state_dict'] = sd

        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

    @torch.no_grad()
    def score_quality(self, x_data, grad_S_fn, t_eval=0.01):
        """Direct score-quality diagnostic: compare model score with −∇S.

        Args:
            x_data:    Reference samples (N, 1, L, L, L) normalised to [−1, 1].
            grad_S_fn: Function x → ∂S/∂x in normalised space (N, 1, L, L, L).
            t_eval:    Time at which model score is evaluated.

        Returns:
            dict with per-sample metrics:
                cos_sim, rel_mse, mag_ratio
        """
        self.eval()
        device = self.device
        x = x_data.to(device)
        N = x.shape[0]

        batch_t = torch.ones(N, device=device) * t_eval
        s_model = self(x, batch_t)
        s_true = -grad_S_fn(x)

        s_m = s_model.reshape(N, -1)
        s_t = s_true.reshape(N, -1)

        dot = (s_m * s_t).sum(dim=1)
        norm_m = s_m.norm(dim=1)
        norm_t = s_t.norm(dim=1)

        cos_sim = dot / (norm_m * norm_t + 1e-12)
        rel_mse = (s_m - s_t).pow(2).sum(dim=1) / (s_t.pow(2).sum(dim=1) + 1e-12)
        mag_ratio = norm_m / (norm_t + 1e-12)

        return {
            'cos_sim': cos_sim.cpu(),
            'rel_mse': rel_mse.cpu(),
            'mag_ratio': mag_ratio.cpu(),
        }

    def _build_time_steps(self, num_steps, eps, schedule, device):
        """Build decreasing time sequence from 1 to eps.

        Args:
            schedule:
                'linear'    – uniform spacing.
                'quadratic' – t = (1-s)²,  mild refinement near 0.
                'cosine'    – cosine map,   smoothly denser near t → 0.
                'log'       – log-uniform,  strong refinement near 0.
                'power_N'   – t = (1-s)^N   (e.g. 'power_4').

        Returns:
            Tensor of shape (num_steps + 1,), monotonically decreasing.
        """
        s = torch.linspace(0, 1, num_steps + 1, device=device)
        if schedule == 'quadratic':
            time_steps = (1.0 - eps) * (1 - s) ** 2 + eps
        elif schedule == 'cosine':
            time_steps = (1.0 - eps) * 0.5 * (1 + torch.cos(s * np.pi)) + eps
        elif schedule == 'log':
            time_steps = torch.exp(
                torch.linspace(0.0, np.log(eps), num_steps + 1, device=device)
            )
        elif schedule.startswith('power_'):
            alpha = float(schedule.split('_')[1])
            time_steps = (1.0 - eps) * (1 - s) ** alpha + eps
        else:  # linear
            time_steps = (1.0 - eps) * (1 - s) + eps
        return time_steps

    @torch.no_grad()
    def sample(self, num_samples=64, num_steps=500, eps=1e-5, schedule='linear'):
        """Euler-Maruyama sampler for 3D data."""
        self.eval()
        device = self.device

        with self.ema.average_parameters(), torch.autocast(device.type, dtype=torch.bfloat16):
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)

            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(num_samples, 1, self.L, self.L, self.L, device=device) * init_std

            for i in tqdm(range(num_steps), desc="Sampling (EM)"):
                time_step = time_steps[i]
                dt = time_steps[i] - time_steps[i + 1]
                batch_t = torch.ones(num_samples, device=device) * time_step
                g = self.diffusion_coeff_fn(time_step)
                mean_x = x + g**2 * self(x, batch_t) * dt
                x = mean_x + g * torch.sqrt(dt) * torch.randn_like(x)
        return mean_x

    def _sigma_to_t(self, sigma_val):
        """Invert marginal_prob_std: given σ, find t such that σ(t) = sigma_val."""
        log_sigma = np.log(self.sigma)
        return np.log(2 * log_sigma * sigma_val**2 + 1) / (2 * log_sigma)

    @torch.no_grad()
    def sample_ode(self, num_samples=64, num_steps=500, eps=1e-5,
                   schedule='log', method='dpm2'):
        """Probability flow ODE sampler via DPM-Solver (deterministic), 3D version."""
        self.eval()
        device = self.device
        with self.ema.average_parameters():
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)
            sigma_steps = torch.stack(
                [self.marginal_prob_std_fn(t) for t in time_steps]).tolist()
            time_steps_f = time_steps.tolist()

            x = torch.randn(num_samples, 1, self.L, self.L, self.L,
                            device=device) * sigma_steps[0]
            batch_t = torch.empty(num_samples, device=device)

            def noise_pred(x, t_val, sigma_t):
                batch_t.fill_(t_val)
                return -sigma_t * self(x, batch_t)

            for i in tqdm(range(num_steps), desc=f"Sampling (DPM-{method[-1]})"):
                sigma_s = sigma_steps[i]
                sigma_next = sigma_steps[i + 1]
                t_s = time_steps_f[i]

                eps_s = noise_pred(x, t_s, sigma_s)

                if method == 'dpm1':
                    x = x + (sigma_next - sigma_s) * eps_s
                elif method == 'dpm2':
                    sigma_mid = (sigma_s * sigma_next) ** 0.5
                    t_mid = self._sigma_to_t(sigma_mid)
                    x_mid = x + (sigma_mid - sigma_s) * eps_s
                    eps_mid = noise_pred(x_mid, t_mid, sigma_mid)
                    x = x + (sigma_next - sigma_s) * eps_mid
                elif method == 'dpm3':
                    s1 = sigma_s ** (2/3) * sigma_next ** (1/3)
                    s2 = sigma_s ** (1/3) * sigma_next ** (2/3)
                    t1 = self._sigma_to_t(s1)
                    t2 = self._sigma_to_t(s2)
                    x1 = x + (s1 - sigma_s) * eps_s
                    e1 = noise_pred(x1, t1, s1)
                    x2 = x + (s2 - sigma_s) * (2 * e1 - eps_s)
                    e2 = noise_pred(x2, t2, s2)
                    x = x + (sigma_next - sigma_s) * (eps_s/6 + 2*e1/3 + e2/6)

        return x

    @torch.no_grad()
    def sample_pc(self, num_samples=64, num_steps=500, snr=0.16, eps=1e-3,
                  corrector_steps=200, schedule='linear'):
        """Predictor-Corrector sampler for 3D data."""
        self.eval()
        device = self.device

        with torch.no_grad():
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)

            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(num_samples, 1, self.L, self.L, self.L, device=device) * init_std

            for i in tqdm(range(num_steps), desc="Sampling (PC)"):
                time_step = time_steps[i]
                dt = time_steps[i] - time_steps[i + 1]
                batch_t = torch.ones(num_samples, device=device) * time_step

                # Corrector (Langevin MCMC)
                for _ in range(corrector_steps):
                    grad = self(x, batch_t)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 0.02 * 2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

                # Predictor (Euler-Maruyama)
                g = self.diffusion_coeff_fn(time_step)
                x_mean = x + g**2 * self(x, batch_t) * dt
                x = x_mean + g * torch.sqrt(dt) * torch.randn_like(x)

        return x_mean

    @torch.no_grad()
    def sample_mala(self, num_samples=64, num_steps=500, t_mh=0.01,
                    mh_steps=20, eps=1e-3, action_fn=None, schedule='linear',
                    mala_step_size=None):
        """Two-phase sampler: EM reverse SDE + MALA refinement.

        Phase 1: EM from t=1 → eps.
        Phase 2: MALA at fixed t=t_mh using the true action_fn.

        Args:
            num_steps:  EM discretisation steps.
            t_mh:       Fixed time at which model score is evaluated in MALA.
            mh_steps:   Number of MALA iterations after EM.
            action_fn:  S(φ) so that π(φ) ∝ exp(−S). Required.
            schedule:   'linear', 'quadratic', 'cosine', 'log', or 'power_N'.
            mala_step_size: Langevin step h.  Default = auto (1.0 / L³).
        """
        assert action_fn is not None, "action_fn is required for MALA"
        self.eval()
        device = self.device

        if mala_step_size is None:
            mala_step_size = 1.0 / (self.L ** 3)

        with torch.no_grad():
            # --- Phase 1: EM reverse SDE ---
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)
            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(num_samples, 1, self.L, self.L, self.L, device=device) * init_std

            for i in tqdm(range(num_steps), desc="Phase 1  EM"):
                time_step = time_steps[i]
                dt = time_steps[i] - time_steps[i + 1]
                batch_t = torch.ones(num_samples, device=device) * time_step
                g = self.diffusion_coeff_fn(time_step)
                mean_x = x + g**2 * self(x, batch_t) * dt
                x = mean_x + g * torch.sqrt(dt) * torch.randn_like(x)
            x = mean_x

            # --- Phase 2: MALA with action_fn ---
            h = torch.tensor(mala_step_size, device=device, dtype=x.dtype)
            batch_t = torch.ones(num_samples, device=device) * t_mh
            total_accept = torch.zeros(num_samples, device=device)
            log_interval = max(1, mh_steps // 10)

            for j in tqdm(range(mh_steps), desc="Phase 2  MALA"):
                score_x = self(x, batch_t)
                noise = torch.randn_like(x)
                y = x + h * score_x + torch.sqrt(2 * h) * noise

                score_y = self(y, batch_t)
                drift = h * (score_x + score_y)
                log_q = -(0.25 / h) * torch.sum(
                    (torch.sqrt(2 * h) * noise + drift) ** 2, dim=(1, 2, 3, 4)
                ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3, 4))
                log_pi = action_fn(x) - action_fn(y)

                accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
                accept = torch.rand(num_samples, device=device) < accept_prob
                x = torch.where(accept[:, None, None, None, None], y, x)
                total_accept += accept.float()

                if (j + 1) % log_interval == 0:
                    running_rate = (total_accept / (j + 1)).mean().item()
                    tqdm.write(f"  MALA step {j+1}/{mh_steps}  "
                               f"accept = {running_rate:.4f}")

        accept_rate = total_accept / mh_steps
        return x, accept_rate
