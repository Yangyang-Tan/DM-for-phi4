"""
PyTorch Lightning diffusion model with EMA support.
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


class DiffusionModel(pl.LightningModule):
    """Score-based diffusion model with EMA."""

    def __init__(self, score_model, sigma=25.0, lr=1e-3, L=32, ema_decay=0.9999,
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
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)
        std = self.marginal_prob_std_fn(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self(perturbed_x, random_t)
        per_sample_loss = torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3))
        return torch.mean(per_sample_loss), per_sample_loss.detach(), random_t.detach()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, per_sample_loss, random_t = self.loss_fn(x)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss, prog_bar=True, on_step=False, on_epoch=True)

        t_bins = [
            (0.0, 0.2, 'loss_UV'),   # small t → low noise → UV/high-k
            (0.2, 0.8, 'loss_mid'),   # medium t
            (0.8, 1.01, 'loss_IR'),   # large t → high noise → IR/low-k
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

    def _build_time_steps(self, num_steps, eps, schedule, device):
        """Build decreasing time sequence from 1 to eps.

        Args:
            schedule:
                'linear'    – uniform spacing, Δt = const.
                'quadratic' – t = (1-s)²,  Δt ∝ √t   (mild refinement near 0).
                'cosine'    – cosine map,   smoothly denser near t → 0.
                'log'       – log-uniform,  Δt ∝ t    (strong but stable).
                'power_N'   – t = (1-s)^N,  Δt ∝ t^{(N-1)/N}  (e.g. 'power_4').

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
        """Euler-Maruyama sampler.

        Args:
            schedule: 'linear', 'quadratic', 'cosine', 'log', or 'power_N'.
                      Aggressiveness: linear < quadratic(Δt∝√t) < cosine
                      < log(Δt∝t) < power_4(Δt∝t^¾) < power_6 ...
        """
        self.eval()
        device = self.device
        with self.ema.average_parameters(), torch.autocast(device.type, dtype=torch.bfloat16):
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)
            dt_all = time_steps[:-1] - time_steps[1:]

            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(num_samples, 1, self.L, self.L, device=device) * init_std
            batch_t = torch.empty(num_samples, device=device)

            for i in tqdm(range(num_steps), desc="Sampling (EM)"):
                batch_t.fill_(time_steps[i].item())
                g = self.diffusion_coeff_fn(time_steps[i])
                dt = dt_all[i]
                mean_x = x + g**2 * self(x, batch_t) * dt
                x = mean_x + g * torch.sqrt(dt) * torch.randn_like(x)
        return mean_x

    def _sigma_to_t(self, sigma_val):
        """Invert marginal_prob_std: given σ, find t such that σ(t) = sigma_val.

        σ²(t) = (Σ^(2t) - 1) / (2·ln Σ)  where Σ = self.sigma
        ⟹ t = ln(2·ln(Σ)·σ² + 1) / (2·ln Σ)
        """
        log_sigma = np.log(self.sigma)
        return np.log(2 * log_sigma * sigma_val**2 + 1) / (2 * log_sigma)

    @torch.no_grad()
    def sample_ode(self, num_samples=64, num_steps=500, eps=1e-5,
                   schedule='log', method='dpm2'):
        """Probability flow ODE sampler via DPM-Solver (deterministic).

        Uses the change of variable λ = -log σ(t) to eliminate stiffness.
        The exact solution for VE-SDE (α=1) is:
            x_t = x_s - ε_θ · (σ_s - σ_t)
        where ε_θ = -σ(t)·score(x, t) is the noise prediction.

        Args:
            num_samples: Number of samples to generate.
            num_steps:   Number of ODE steps.
            eps:         End time (>0, avoids t=0 singularity).
            schedule:    Time step schedule ('linear', 'log', etc.).
            method:      'dpm1' (1st order, 1 NFE/step),
                         'dpm2' (2nd order midpoint, 2 NFE/step, recommended),
                         'dpm3' (3rd order, 3 NFE/step),
                         or 'rk45' (adaptive scipy solver).
        """
        if method == 'rk45':
            return self._sample_ode_rk45(num_samples, eps)

        self.eval()
        device = self.device
        with self.ema.average_parameters():
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)
            sigma_steps = torch.stack(
                [self.marginal_prob_std_fn(t) for t in time_steps]).tolist()
            time_steps_f = time_steps.tolist()

            x = torch.randn(num_samples, 1, self.L, self.L,
                            device=device) * sigma_steps[0]
            batch_t = torch.empty(num_samples, device=device)

            def noise_pred(x, t_val, sigma_t):
                """ε_θ(x, t) = -σ(t) · score(x, t)"""
                batch_t.fill_(t_val)
                return -sigma_t * self(x, batch_t)

            for i in tqdm(range(num_steps), desc=f"Sampling (DPM-{method[-1]})"):
                sigma_s = sigma_steps[i]
                sigma_next = sigma_steps[i + 1]
                t_s = time_steps_f[i]

                eps_s = noise_pred(x, t_s, sigma_s)

                if method == 'dpm1':
                    # DPM-Solver-1: x_next = x + (σ_next - σ_s) · ε_θ
                    x = x + (sigma_next - sigma_s) * eps_s

                elif method == 'dpm2':
                    # DPM-Solver-2 (midpoint): geometric mean in σ-space
                    sigma_mid = (sigma_s * sigma_next) ** 0.5
                    t_mid = self._sigma_to_t(sigma_mid)
                    x_mid = x + (sigma_mid - sigma_s) * eps_s
                    eps_mid = noise_pred(x_mid, t_mid, sigma_mid)
                    x = x + (sigma_next - sigma_s) * eps_mid

                elif method == 'dpm3':
                    # DPM-Solver-3: intermediates at 1/3 and 2/3 in σ-space
                    sigma_1 = sigma_s ** (2/3) * sigma_next ** (1/3)
                    sigma_2 = sigma_s ** (1/3) * sigma_next ** (2/3)
                    t_1 = self._sigma_to_t(sigma_1)
                    t_2 = self._sigma_to_t(sigma_2)

                    x_1 = x + (sigma_1 - sigma_s) * eps_s
                    eps_1 = noise_pred(x_1, t_1, sigma_1)

                    x_2 = x + (sigma_2 - sigma_s) * (2 * eps_1 - eps_s)
                    eps_2 = noise_pred(x_2, t_2, sigma_2)

                    # Simpson-like combination
                    x = x + (sigma_next - sigma_s) * (
                        eps_s / 6 + 2 * eps_1 / 3 + eps_2 / 6)

        return x

    @torch.no_grad()
    def _sample_ode_rk45(self, num_samples, eps=1e-5):
        """Probability flow ODE with scipy adaptive RK45 solver.

        Most accurate but slow (CPU-GPU transfer at each step).
        Useful as ground-truth reference for validating other methods.
        """
        from scipy import integrate

        self.eval()
        device = self.device
        shape = (num_samples, 1, self.L, self.L)

        with self.ema.average_parameters():
            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x0 = (torch.randn(*shape, device=device) * init_std).cpu().numpy().flatten()

            def ode_func(t, x_flat):
                x = torch.tensor(x_flat, dtype=torch.float32, device=device).reshape(shape)
                batch_t = torch.ones(num_samples, device=device) * t
                g = self.diffusion_coeff_fn(torch.tensor(t, device=device))
                score = self(x, batch_t)
                drift = 0.5 * g**2 * score
                return drift.cpu().numpy().flatten()

            print(f"Running RK45 ODE solver (t: 1.0 → {eps})...")
            solution = integrate.solve_ivp(
                ode_func, (1.0, eps), x0,
                method='RK45', rtol=1e-5, atol=1e-5,
            )
            print(f"  RK45 finished: {solution.nfev} function evaluations")

            x = torch.tensor(
                solution.y[:, -1], dtype=torch.float32, device=device
            ).reshape(shape)

        return x

    @torch.no_grad()
    def sample2(self, num_samples=64, num_steps=500, eps=1e-3):
        """Euler-Maruyama sampler."""
        self.eval()
        device = self.device

        with self.ema.average_parameters():
            time_steps = torch.linspace(1.0, eps, num_steps, device=device)
            x_all = torch.zeros(num_steps, num_samples, 1, self.L, self.L, device=device)
            torch.manual_seed(1234)
            # x = torch.randn(num_samples, 1, self.L, self.L, device=device) * init_std
            x = torch.randn(num_samples, 1, self.L, self.L, device=device)
            for time_idx, time_step in tqdm(enumerate(time_steps), desc="Sampling (EM)"):
                x_all[time_idx, :, :, :, :] = x
                batch_t = torch.ones(num_samples, device=device) * torch.tensor(0.2, device=device)
                # g = 0.05*self.diffusion_coeff_fn(time_step)
                mean_x = x + self(x, batch_t) * torch.tensor(0.02, device=device)
                x = mean_x + torch.sqrt(2*torch.tensor(0.02, device=device)) * torch.randn_like(x)
        return x_all

    @torch.no_grad()
    def sample_pc(self, num_samples=64, num_steps=500, snr=0.16, eps=1e-3,
                  corrector_steps=200, schedule='linear'):
        """Predictor-Corrector sampler.

        Args:
            schedule: 'linear', 'quadratic', 'cosine', 'log', or 'power_N'.
        """
        self.eval()
        device = self.device

        with self.ema.average_parameters(), torch.autocast(device.type, dtype=torch.bfloat16):
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)
            dt_all = time_steps[:-1] - time_steps[1:]

            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(num_samples, 1, self.L, self.L, device=device) * init_std
            batch_t = torch.empty(num_samples, device=device)
            noise_norm = np.sqrt(np.prod(x.shape[1:]))

            for i in tqdm(range(num_steps), desc="Sampling (PC)"):
                batch_t.fill_(time_steps[i].item())
                dt = dt_all[i]

                # Corrector (Langevin MCMC)
                for _ in range(corrector_steps):
                    grad = self(x, batch_t)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    langevin_step_size = 0.02*2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

                # Predictor (Euler-Maruyama)
                g = self.diffusion_coeff_fn(time_steps[i])
                x_mean = x + g**2 * self(x, batch_t) * dt
                x = x_mean + g * torch.sqrt(dt) * torch.randn_like(x)

        return x_mean

    @torch.no_grad()
    def sample_mala(self, num_samples=64, num_steps=500, t_mh=0.01,
                    mh_steps=20, eps=1e-3, action_fn=None, schedule='linear',
                    mala_step_size=None):
        """Two-phase sampler: EM reverse SDE + MALA refinement.

        Phase 1: EM from t=1 → eps  (same as sample()).
        Phase 2: MALA at fixed t=t_mh using the true action_fn.

        Args:
            num_steps:  EM discretisation steps.
            t_mh:       Fixed time at which model score is evaluated in MALA.
            mh_steps:   Number of MALA iterations after EM.
            action_fn:  S(φ) so that π(φ) ∝ exp(−S). Required.
            schedule:   'linear', 'quadratic', 'cosine', 'log', or 'power_N'.
            mala_step_size: Langevin step h.  Default = auto (1.0 / L²).

        Returns:
            x:           Final samples  (num_samples, 1, L, L).
            accept_rate: Per-sample acceptance rate  (num_samples,).
        """
        assert action_fn is not None, "action_fn is required for MALA diagnostic"
        self.eval()
        device = self.device

        if mala_step_size is None:
            mala_step_size = 1.0 / (self.L ** 2)

        with self.ema.average_parameters(), torch.autocast(device.type, dtype=torch.bfloat16):
            # --- Phase 1: EM reverse SDE ---
            time_steps = self._build_time_steps(num_steps, eps, schedule, device)
            dt_all = time_steps[:-1] - time_steps[1:]
            init_std = self.marginal_prob_std_fn(torch.tensor(1.0, device=device))
            x = torch.randn(num_samples, 1, self.L, self.L, device=device) * init_std
            batch_t = torch.empty(num_samples, device=device)

            for i in tqdm(range(num_steps), desc="Phase 1  EM"):
                batch_t.fill_(time_steps[i].item())
                dt = dt_all[i]
                g = self.diffusion_coeff_fn(time_steps[i])
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
                    (torch.sqrt(2 * h) * noise + drift) ** 2, dim=(1, 2, 3)
                ) + 0.5 * torch.sum(noise ** 2, dim=(1, 2, 3))
                log_pi = action_fn(x) - action_fn(y)

                accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
                accept = torch.rand(num_samples, device=device) < accept_prob
                x = torch.where(accept[:, None, None, None], y, x)
                total_accept += accept.float()

                if (j + 1) % log_interval == 0:
                    running_rate = (total_accept / (j + 1)).mean().item()
                    tqdm.write(f"  MALA step {j+1}/{mh_steps}  "
                               f"accept = {running_rate:.4f}")

        accept_rate = total_accept / mh_steps
        return x, accept_rate

    @torch.no_grad()
    def score_quality(self, x_data, grad_S_fn, t_eval=0.01):
        """Direct score-quality diagnostic: compare model score with −∇S.

        No MALA, no step-size tuning — just a single forward pass.

        Args:
            x_data:    Reference samples (N, 1, L, L) normalised to [−1, 1].
            grad_S_fn: Function x → ∂S/∂x in normalised space (N, 1, L, L).
            t_eval:    Time at which model score is evaluated.

        Returns:
            dict with per-sample metrics:
                cos_sim:    cosine similarity  ⟨s_model, −∇S⟩ / (|s|·|∇S|)
                rel_mse:    ||s_model − (−∇S)||² / ||∇S||²
                mag_ratio:  |s_model| / |∇S|
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

    @torch.no_grad()
    def denoising_score_eval(self, x_data, t_values=None):
        """Evaluate denoising score matching loss at multiple time points.

        Adds noise to clean data and checks if model correctly predicts
        score ≈ −z/σ(t). Valid at any t, no knowledge of −∇S needed.

        Args:
            x_data:   Clean samples (N, 1, L, L) normalised to [−1, 1].
            t_values: List of time points to evaluate. Defaults to
                      [0.01, 0.05, 0.1, 0.2, 0.5, 0.8].

        Returns:
            dict mapping t → mean denoising loss  E[||s·σ + z||²]
        """
        if t_values is None:
            t_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
        self.eval()
        device = self.device
        x = x_data.to(device)
        N = x.shape[0]
        results = {}
        for t_val in t_values:
            z = torch.randn_like(x)
            std = self.marginal_prob_std_fn(torch.tensor(t_val, device=device))
            x_t = x + z * std
            batch_t = torch.ones(N, device=device) * t_val
            s = self(x_t, batch_t)
            loss = ((s * std + z) ** 2).sum(dim=(1, 2, 3)).mean().item()
            results[t_val] = loss
        return results
