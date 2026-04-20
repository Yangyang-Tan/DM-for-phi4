# CLAUDE.md

## Overview

Score-based diffusion models for 2D/3D scalar phi^4 lattice field theory and image generation (MNIST, CIFAR-10, CelebA, STL-10, ChestMNIST). Networks learn the score function; samples via reverse SDE/ODE. Physics params: hopping `kappa`, coupling `lambda`.

**Active research thread (2026-04):** image-diffusion propagator studies — train on CelebA/CIFAR-10/STL-10 with log-scale checkpoints, then sweep many epochs with SDE + ODE (DPM) samplers and measure propagator evolution in `correlation_*.jl`. Also: 2Dphi4 `sigma` ablation.

## Stack

- **Python** (conda env `nenv2`): PyTorch Lightning, `torch_ema`, `h5py`. Deps in `requirements.txt`.
  - Run as: `conda run --no-capture-output -n nenv2 python …` (do NOT try to `source conda.sh` — path guessing has failed before).
- **Julia**: Wolff cluster + Fourier-Accelerated HMC for training data + lattice correlators. Packages: `FFTW`, `JLD2`, `ProgressMeter`.

## Layout

> `.gitignore` hides binary data (`.npy/.jld2/.ckpt/.pt`), outputs, LaTeX build. Use `ls` for the full tree.

### Root (shared 2D core)
- `diffusion_lightning.py` — `DiffusionModel` (Lightning): loss, EMA, samplers (EM, PC, MALA, DPM1/2/3, RK45)
- `networks.py` — 2D score nets: `ScoreNet`, `ScoreNetUNet`, `ScoreNetUNetPeriodic`, `NCSNpp2D`
- `data.py` — `MNISTDataModule`, `FieldDataModule` (HDF5/JLD2 with `cfgs` key), `GPUDataLoader`
- `WolffFAHMC_ND.jl` — D-dim Wolff+FAHMC used by both 2D/3D training-data generators

### Domain dirs
- `2Dphi4/` — 2D phi^4 (`train_phi4.py`, `sample_phi4.py`, analysis, Julia scripts). See `2Dphi4/CLAUDE.md`.
- `3Dphi4/` — 3D phi^4; **local** `diffusion_lightning_3d.py`, `networks_3d.py`, `data_3d.py`. See `3Dphi4/CLAUDE.md`.
- `celeba/` — CelebA 64/128 grayscale. `train_celeba.py`, `sample_celeba.py`, `sample_at_t0_celeba.py`, `celeba_datamodule.py`, `correlation_celeba.jl`, `preprocess_celeba.py`, `batch_sample_celeba.sh`. Data: `celeba_gray{64,128}.npy`.
- `cifar10/` — `train_cifar10.py`, `sample_cifar10.py`, `sample_at_t0_cifar10.py`, `cifar10_datamodule.py` (class filtering), `correlation_cifar10.jl`.
- `stl10/` — 64×64 grayscale (resized from 96 via `preprocess_stl10.py`). `train_stl10.py`, `sample_stl10.py`, `sample_at_t0_stl10.py`, `stl10_datamodule.py`, `correlation_stl10.jl`. Data: `stl10_gray64_{unlabeled,train,test}.npy`.
- `MNIST/`, `medmnist/` — image training/sampling.
- `models/`, `references/` — checkpoints, reference implementations.

## Commands

```bash
# Training (from the domain dir)
python train_phi4.py --L 128 --k 0.5 --l 0.022 --network ncsnpp --device cuda:0        # 2Dphi4/
python train_phi4.py --L 64  --k 0.2 --l 0.9   --network ncsnpp --device cuda:0        # 3Dphi4/ (bf16 + torch.compile)
python train_celeba.py --size 128 --network ncsnpp --device cuda:0                     # celeba/ (log-scale ckpts)
python train_stl10.py  --network ncsnpp --device cuda:0                                # stl10/

# Sampling — methods: em, pc, mala, dpm1, dpm2, dpm3, rk45
python sample_phi4.py   --L 128 --k 0.5 --l 0.022 --network ncsnpp --method em  --num_samples 1024 --num_steps 1000 --schedule log
python sample_celeba.py --size 128 --ep "epoch=1099" --method em --num_steps 2000 --schedule log --num_samples 512 --seed 42
python sample_at_t0_celeba.py  ...   # snapshots at multiple t0 for IR→UV propagator convergence

# Training data (Julia)
julia WolffFAHMC.jl    # from 2Dphi4/trainingdata/ or 3Dphi4/trainingdata/
```

Typical sampling flags: `--method em --num_steps 2000 --schedule log --num_samples 512 --seed <fixed> --n_repeats 1`; ODE (DPM) uses `--num_steps 100–400`.

## Architecture

- **SDE**: Variance-Exploding with `sigma^t`. Network output divided by `marginal_prob_std(t)`.
- **`sigma` is NOT Song's `σ_max`.** This codebase parameterizes VE-SDE so `std(t=1) = sqrt((σ²−1)/(2·ln σ))`. Never set `sigma` equal to the max pairwise data distance — match `std(t=1)` to Song's `σ_max` criterion instead.
- **Periodic BC**: phi^4 nets use `padding_mode="circular"`. Use periodic variants for phi^4, standard variants for images.
- **Time conditioning**: Gaussian Fourier projection, added into each conv block.
- **NCSNpp**: ResNet blocks + time cond, U-Net skips. `ResnetBlock` has NO residual (removed for stability); `skip_conv` retained only for ckpt compatibility.
- **2D vs 3D**: 3D mirrors 2D but `Conv3d`, 5D tensors `[B,C,D,H,W]`, `DiffusionModel3D`. 3D uses `torch.compile` + bf16-mixed.
- **Normalization**: Fields → [-1,1] via global min/max; `norm_min/norm_max` stored in ckpt. 3D caches to `.norm.json`.
- **EMA**: `torch_ema`, configurable start epoch.
- **Time schedules**: `linear`, `quadratic`, `cosine`, `log`, `power_N`. Use `log` for low-noise regime (concentrates steps near t=0).
- **MALA sampler**: Phase 1 reverse SDE (EM) → Phase 2 Metropolis-adjusted Langevin with true phi^4 action.
- **Loss bins**: UV (t<0.2), mid (0.2–0.8), IR (t>0.8).

## GPU

- Env: `nenv2`
- CUDA mapping (PyTorch; differs from nvidia-smi):
  - `cuda:0,1` → RTX 4090 (24GB)
  - `cuda:2,3` → RTX 5090 (32GB)
  - `cuda:4,5` → RTX 2080 Ti (11GB)
- `torch.compile` adds `_orig_mod.` prefix. Both `DiffusionModel` (2D) and `DiffusionModel3D` (3D) strip/add it in `on_load_checkpoint` — no manual fixup in sampling scripts.

## Checkpoints

- **Log-scale saving** (preferred for new training scripts): `LogScaleCheckpoint` callback saves ~50–100 ckpts geometrically spaced — use this in new image/phi4 training scripts, not fixed-interval saves.
- **Naming quirks** — match the `--ep` flag to the on-disk filename verbatim:
  - `celeba_64_ncsnpp/models/epoch=0001.ckpt` → `--ep "epoch=0001"`
  - `celeba_128_ncsnpp/models/epoch=epoch=0000.ckpt` (double-prefixed!) → `--ep "epoch=epoch=0000"`
- **`Missing key(s) in state_dict: "score_model.embed.0.W"`** on load → sampler instantiated wrong network class (`ScoreNetUNet` vs `NCSNpp2D`) or wrong `image_size`. Fix by matching `--network` and `--image_size` to the ckpt; `DiffusionModel.load_from_checkpoint` needs the exact `score_model=<class>` passed in.
- **Filenames**: `diffusion_L{L}_k{k}_l{l}*.ckpt` or `phi4_L{L}_k{k}_l{l}_{network}/models/epoch=NNNN.ckpt`.

## Paper

- `draft/DM.tex` — revtex4-2, two-column. `\graphicspath{{./figures/}{./figures/3Dphi4/}}`.
- Figures: PDF/vector. Sizes: `(10,4)` for 1×2 calibration, `(7,4.9)` for acceptance, `(10,7)` for 2×2 loss decomposition.
- Compile from `draft/`: `pdflatex -interaction=nonstopmode DM.tex` (run twice).

## Data

- HDF5/JLD2 with `cfgs` key. Filename: `cfgs_k={kappa}_l={lambda}_{L}^{D}_t={thermalization}.jld2`
- Image data as `.npy` (e.g. `celeba_gray64.npy`, `stl10_gray64_unlabeled.npy`).
- 2D/3D train scripts import from root via `sys.path.append("..")`.
