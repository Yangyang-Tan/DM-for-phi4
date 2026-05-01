# 2D phi^4 Lattice Field Theory

2D scalar phi^4 theory diffusion model training, sampling, and analysis. Lattice sizes L=8,16,32,64,128; hopping parameter kappa=0.26,0.2705,0.28; coupling lambda=0.022.

## Key Scripts

- `train_phi4.py` — Training script. Run: `python train_phi4.py --L 128 --k 0.5 --l 0.022 --network ncsnpp --device cuda:0`
- `train_phi4_multiL.py` — Multi-L joint training
- `sample_phi4.py` / `sample_phi4_sweep.py` / `sample_phi4_crossL.py` — Sampling (EM/PC/MALA, multi-ckpt sweep, cross-L)
- `sample_at_t0.py` — Sample evaluation at t=0
- `analysis/plot_loss_curves.py` — Loss decomposition plotting
- All other data-analysis scripts live under `analysis/`

## Import Pattern

All Python scripts use `sys.path.append("..")` to import root-level modules:
- `diffusion_lightning` (DiffusionModel)
- `networks` (ScoreNet, ScoreNetUNet, NCSNpp2D, etc.)
- `data` (FieldDataModule)

## Julia Scripts

Top-level (simulator entry points; read/write `trainingdata/`, `data/`):
- `Langevin.jl` — Langevin dynamics simulation (GPU-accelerated)
- `Phi_4_model.jl` — phi^4 model definitions

Under `julia/` (analysis/utility, run from `2Dphi4/` as `julia julia/<file>.jl`):
- `correlation_2D.jl` — Two-point correlation function measurement
- `CorrelationUtils.jl` / `CorrelationUtilsGPU.jl` — Shared correlation utilities
- `cumulant.jl` — Higher-order cumulant computation
- `plot.jl` — Plotting helpers

## Subdirectories

- `analysis/` — Python data-analysis scripts (cross-L propagator, σ rule check, ξ extraction, loss curves, sweep diagnostics, etc.). When run from `2Dphi4/`, scripts here use `Path(__file__).resolve().parent` to import sibling analysis modules.
- `acceptance_rate/` — MALA acceptance rate analysis: `calibrate_step_size.py`, `calibrate_t_mh.py`, `acceptance_vs_epoch.py`, `acceptance_scan.py`
- `score_quality_analysis/` — Score quality metrics by kappa and L (includes `legacy_top/` snapshot of an older outputs version)
- `scripts/` — Shell drivers (`run_sigma_ablation*.sh`, `run_sweep_samples*.sh`). Invoke from `2Dphi4/` via `bash scripts/<name>.sh`; `LOG_DIR` writes under `results/sigma_ablation/...`.
- `julia/` — Julia analysis/utility scripts (see above)
- `mathematica/` — `*.wls` plotting scripts
- `results/` — All run/sweep outputs (gitignored aside from a few committed PDFs):
  - `crossL/` — cross-L propagator/sample logs and PDFs (formerly `crossL_logs/`)
  - `train_logs/` — multi-L training logs
  - `sigma_ablation/{all,L32,L32_k26,L32_k2705,L32_compare}/` — σ-ablation logs and figures (formerly `sigma_ablation*_logs/`)
  - `sigma_comparison_L128/` — L=128 sweep diagnostics + propagator caches (committed)
- `trainingdata/` — Julia-generated .jld2 training data (gitignored, ~26 GB)
- `runs/` — All training output directories (gitignored). Each is `runs/phi4_L{L}_k{k}_l{λ}_ncsnpp[_sigma{σ}]/` with `models/`, `data/`, and TensorBoard `lightning_logs/`. Multi-L runs: `runs/phi4_Lmulti{Ls}_k{k}_l{λ}_ncsnpp[_lcond]/`. Train/sample/analysis scripts hardcode the `runs/` prefix.
