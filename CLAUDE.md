# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.


## 3. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## Overview

Score-based diffusion models for 2D/3D scalar phi^4 lattice field theory and image generation (MNIST, CIFAR-10, CelebA, STL-10, ChestMNIST). Networks learn the score function; samples via reverse SDE/ODE. Physics params: hopping `kappa`, coupling `lambda`.

**Active research thread (2026-04):** image-diffusion propagator studies — train on CelebA/CIFAR-10/STL-10 with log-scale checkpoints, then sweep many epochs with SDE + ODE (DPM) samplers and measure propagator evolution in `correlation_*.jl`. Also: 2Dphi4 `sigma` ablation.

## Stack

- **Python** (conda env `nenv2`): PyTorch Lightning, `torch_ema`, `h5py`. Deps in `requirements.txt`.

Typical sampling flags: `--method em --num_steps 2000 --schedule log --num_samples 512 --seed <fixed> --n_repeats 1`; ODE (DPM) uses `--num_steps 100–400`.

## Architecture

- **SDE**: Variance-Exploding with `sigma^t`. Network output divided by `marginal_prob_std(t)`.
- **`sigma` is NOT Song's `σ_max`.** This codebase parameterizes VE-SDE so `std(t=1) = sqrt((σ²−1)/(2·ln σ))`.
- **Empirical sigma rule (L=128 ablation, validated 2026-04):** choose σ such that `σ_max ≈ 2·std(t=1)` — equivalently `std(t=1) ≈ σ_max/2` — where `σ_max` is the max pairwise Euclidean distance of training cfgs in **original (un-normalized) data space**. The normalized-space variant is off by ~5× and wrong. Verified at L=128: predicted σ matches the ablation winner to within ≤15% (k=0.2705 → predict 479, best 450; k=0.28 → predict 735, best 640). Reusable check: [2Dphi4/analysis/check_sigma_rule.py](2Dphi4/analysis/check_sigma_rule.py).

## GPU
- CUDA mapping (PyTorch; differs from nvidia-smi):
  - `cuda:0,1` → RTX 4090 (24GB)
  - `cuda:2,3` → RTX 5090 (32GB)
  - `cuda:4,5` → RTX 2080 Ti (11GB)
- **Device-selection policy:**
  - Prefer `cuda:2` and `cuda:3` (5090, 32GB) for new jobs.
  - Fall back to `cuda:0` and `cuda:1` (4090, 24GB) when 2/3 are busy.
  - Avoid `cuda:4` and `cuda:5` (2080 Ti, 11GB) unless explicitly requested — small VRAM, much slower.
  - **One job per GPU.** Before launching, check `nvidia-smi` and pick a GPU that is currently idle; never co-locate two compute jobs on the same device (OOM and contention are routine on this box).
- `torch.compile` adds `_orig_mod.` prefix. Both `DiffusionModel` (2D) and `DiffusionModel3D` (3D) strip/add it in `on_load_checkpoint` — no manual fixup in sampling scripts.

## Checkpoints

- **Log-scale saving** : `LogScaleCheckpoint` callback saves ~50–100 ckpts geometrically spaced
