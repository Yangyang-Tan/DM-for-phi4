# DM Project Structure

This repository keeps active code paths stable and groups inactive material away
from the root directory.

## Core Code

- `data.py`: shared Lightning data modules for MNIST and 2D/3D field data.
- `diffusion_lightning.py`: shared 2D/3D diffusion Lightning module and samplers.
- `phi4_action.py`: shared 2D/3D phi4 action and gradient.
- `networks_nd.py`: shared dimension-agnostic 2D/3D lattice network blocks.
- `networks.py`: public 2D/image network API, kept for existing scripts.
- `3Dphi4/networks_3d.py`: public 3D network API, kept for existing scripts.

## Experiments

- `2Dphi4/`: 2D phi4 training, sampling, analysis, and result folders.
- `3Dphi4/`: 3D phi4 training, sampling, analysis, and result folders.
- `MNIST/`, `cifar10/`, `celeba/`, `stl10/`, `medmnist/`: image-diffusion
  experiments and dataset-specific utilities.

## Generated Or Large Outputs

- `data/`, `models/`, `runs/`, and per-experiment `*_logs/`, `data/`,
  `models/`, `figures/` folders are output locations used by existing scripts.
  They are intentionally not moved because many CLI defaults refer to them.

## Papers And Old Work

- `draft/`: active paper draft and paper figures.
- `overleaf/`: local Overleaf git checkout, ignored by the main repository.
- `docs/archive/`: old notebooks/scripts kept for reference.
- `docs/proposal/`: proposal and CV TeX/PDF material.
- `references/`: third-party reference implementations, ignored by the main
  repository.
