# [CLAUDE.md](http://CLAUDE.md)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Score-based diffusion models for lattice field theory (phi^4 theory) and image generation. The project trains neural networks to learn the score function (gradient of log-probability) of field configurations, then generates new samples via reverse SDE integration. Two main physics targets: 2D and 3D scalar phi^4 lattice field theory at various hopping parameters (kappa) and coupling constants (lambda).

## Language and Framework

- **Python**: PyTorch Lightning for training/sampling diffusion models. Dependencies listed in `requirements.txt`. Conda environment: `nenv2`.
- **Julia**: Lattice Monte Carlo simulations (Wolff cluster + Fourier-Accelerated HMC) for generating training data. Key packages: `FFTW`, `JLD2`, `ProgressMeter`.

## Repository Structure

> **Note:** `.gitignore` hides all binary data (`.npy`, `.jld2`, `.ckpt`, `.pt`), training outputs, and LaTeX build artifacts. Only source code files (`.py`, `.jl`, `.tex`, `.bib`, `.sh`) are visible to code analysis tools. Use `ls` directly to see the full directory listing.

### Shared core (root level)

- `diffusion_lightning.py` — 2D `DiffusionModel` (Lightning module): loss, EMA, samplers (EM, PC, MALA)
- `networks.py` — 2D score network architectures: `ScoreNet` (no downsampling), `ScoreNetUNet` (standard U-Net), `ScoreNetUNetPeriodic` (periodic BC U-Net), `NCSNpp2D` (NCSN++ with ResNet blocks)
- `data.py` — DataModules: `MNISTDataModule`, `FieldDataModule` (loads HDF5/JLD2 with `cfgs` key), `GPUDataLoader`
- `WolffFAHMC_ND.jl` — D-dimensional Wolff cluster + FAHMC sampler, used by both 2D and 3D training data generators

### Domain-specific directories

- `2Dphi4/` — 2D phi^4: `train_phi4.py`, `sample_phi4.py`, analysis scripts, Julia correlation/Langevin scripts. See `2Dphi4/CLAUDE.md`.
- `3Dphi4/` — 3D phi^4: contains **local** 3D modules (`diffusion_lightning_3d.py`, `networks_3d.py`, `data_3d.py`) plus `train_phi4.py`, `sample_phi4.py`. See `3Dphi4/CLAUDE.md`.
- `MNIST/` — MNIST training/sampling scripts
- `cifar10/` — CIFAR-10 grayscale training (`cifar10_datamodule.py` with class filtering)
- `medmnist/` — ChestMNIST training/sampling
- `models/` — Saved checkpoints (naming: `diffusion_L{L}_k{k}_l{l}*.ckpt`)
- `references/` — Reference implementations (DMasSQ, score_sde_pytorch, etc.)

## Common Commands

### Training

```bash
# 2D phi^4 (from 2Dphi4/)
python train_phi4.py --L 128 --k 0.5 --l 0.022 --network ncsnpp --device cuda:0

# 3D phi^4 (from 3Dphi4/, uses bf16-mixed precision and torch.compile)
python train_phi4.py --L 64 --k 0.2 --l 0.9 --network ncsnpp --device cuda:0

# MNIST (from MNIST/)
python train_mnist.py --epochs 5000 --device cuda:0

# ChestMNIST (from medmnist/)
python train_chestmnist.py --size 64 --device cuda:0
```

### Sampling

```bash
# 2D phi^4 (from 2Dphi4/)
python sample_phi4.py --L 128 --k 0.5 --l 0.022 --network ncsnpp --method em --num_samples 1024 --num_steps 1000

# Sampling methods: em (Euler-Maruyama), pc (Predictor-Corrector), mala (EM + MALA refinement)
```

### Generating training data (Julia)

```bash
# From 2Dphi4/trainingdata/ or 3Dphi4/trainingdata/
julia WolffFAHMC.jl
```

## Architecture Notes

- **SDE framework**: Variance-Exploding SDE with `sigma^t` diffusion coefficient. Score is normalized by `marginal_prob_std(t)` at the network output.
- **Periodic boundary conditions**: Lattice field theory networks use `padding_mode="circular"` to respect lattice periodicity. This is critical — use periodic variants for phi^4, standard variants for images.
- **Time conditioning**: Gaussian Fourier projection for time embedding, injected additively into each conv block.
- **NCSNpp variants**: ResNet blocks with time conditioning, U-Net with skip connections at every block. Note: `ResnetBlock` intentionally has no residual connection (removed for training stability; `skip_conv` exists only for checkpoint compatibility).
- **2D vs 3D**: 3D modules (`diffusion_lightning_3d.py`, `networks_3d.py`, `data_3d.py`) live in `3Dphi4/` (not root). They mirror 2D but use `Conv3d`, 5D tensors `[B, C, D, H, W]`, and `DiffusionModel3D`. The 3D training script also uses `torch.compile` and bf16 mixed precision.
- **Normalization**: Field data is normalized to [-1, 1] using global min/max. `norm_min`/`norm_max` are saved in checkpoints for denormalization during sampling. 3D data caches norm params to `.norm.json` files.
- **EMA**: Exponential moving average of score model parameters via `torch_ema`. Configurable start epoch.
- **Time step schedules**: Samplers support `linear`, `quadratic`, `cosine`, `log`, and `power_N` schedules for time discretization. `log` concentrates steps near t=0 (important for low-noise regime).
- **MALA sampler**: Two-phase approach — Phase 1 runs reverse SDE (EM), Phase 2 refines with Metropolis-adjusted Langevin using the true phi^4 action as target.
- **Loss binning**: Training loss is decomposed into UV (t<0.2), mid (0.2-0.8), and IR (t>0.8) bins for diagnostics.
<!-- 
## MALA Acceptance Rate Diagnostic

Analysis scripts for measuring MALA acceptance rate as a training diagnostic:

### Scripts

- `2Dphi4/calibrate_step_size.py` — Sweep coefficient `c` to calibrate step size `h=c/L^2`
- `2Dphi4/calibrate_t_mh.py` — Sweep evaluation time `t_mh`
- `2Dphi4/acceptance_vs_epoch.py` — Track acceptance rate vs training epoch
- `2Dphi4/acceptance_scan.py` — Scan all (L, kappa) combinations
- `3Dphi4/calibrate_step_size.py` — 3D version, step size `h=c/L^3`
- `3Dphi4/calibrate_t_mh.py` — 3D version
- `3Dphi4/acceptance_vs_epoch.py` — 3D version

### Key parameters

- **Step size scaling**: `h = c/L^D` (NOT Roberts & Rosenthal's `d^{-1/3}` — phi^4 nearest-neighbor coupling violates the i.i.d. assumption)
- **Universal coefficient**: `c ≈ 0.2` optimal across both 2D and 3D
- **Evaluation time**: `t_mh = 1e-4` optimal for diagnostic power
- **Single MH step**: Avoids warm-up/selection bias
- **Mini-batch inference for 3D**: `batch_size=128` to avoid OOM on 32GB GPUs

### Running examples

```bash
# 2D calibration (from 2Dphi4/)
python calibrate_step_size.py --device cuda:0 --L 128 --k 0.28 --num_samples 1024
python calibrate_t_mh.py --device cuda:0 --L 128 --k 0.28 --c 0.5 --num_samples 1024

# 2D acceptance vs epoch
python acceptance_vs_epoch.py --device cuda:0 --L 128 --k 0.2705 --every 100 --c 0.2 --num_samples 1024 --mh_steps 1

# 3D calibration (from 3Dphi4/, needs ≥24GB GPU)
python calibrate_step_size.py --device cuda:2 --L 64 --k 0.2 --num_samples 512
python calibrate_t_mh.py --device cuda:2 --L 64 --k 0.2 --c 0.2 --num_samples 512

# 3D acceptance vs epoch
python acceptance_vs_epoch.py --device cuda:2 --L 64 --k 0.2 --l 0.9 --num_samples 512 --every 100 --c 0.2 --mh_steps 1
``` -->

## GPU Environment

- **Conda environment**: `nenv2` (has pytorch, torch_ema, pytorch_lightning, h5py)
- **CUDA device mapping** (PyTorch numbering, differs from nvidia-smi):
  - `cuda:0`, `cuda:1` — RTX 4090 (24GB)
  - `cuda:2`, `cuda:3` — RTX 5090 (32GB)
  - `cuda:4`, `cuda:5` — RTX 2080 Ti (11GB)
- **3D models require ≥24GB** (4090 or 5090); 2D models fit on 4090
- **torch.compile** adds `_orig_mod.` prefix to state dict keys (handled via BytesIO workaround in 3D scripts)

## Paper Draft

- `draft/DM.tex` — Main LaTeX file (revtex4-2, two-column)
- `draft/figures/` — All figures, organized by `2Dphi4/` and `3Dphi4/` subdirectories
- `\graphicspath{{./figures/}{./figures/3Dphi4/}}` set in preamble
- **Figure format**: All acceptance/calibration figures output as PDF (vector); use `figsize=(10, 4)` for 1×2 calibration panels, `figsize=(7, 4.9)` for single acceptance plots, `figsize=(10, 7)` for 2×2 loss decomposition
- Compile from `draft/` directory: `pdflatex -interaction=nonstopmode DM.tex` (run twice for references)

## Data Format

- Training data stored as HDF5/JLD2 files with a `cfgs` key containing field configurations.
- Naming convention: `cfgs_k={kappa}_l={lambda}_{L}^{D}_t={thermalization}.jld2`
- Checkpoint naming: `diffusion_L{L}_k{k}_l{l}*.ckpt` or `phi4_L{L}_k{k}_l{l}_{network}/models/epoch=NNNN.ckpt`
- 2D/3D training scripts import from root via `sys.path.append("..")`.
