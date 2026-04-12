# 2D phi^4 Lattice Field Theory

2D scalar phi^4 theory diffusion model training, sampling, and analysis. Lattice sizes L=8,16,32,64,128; hopping parameter kappa=0.26,0.2705,0.28; coupling lambda=0.022.

## Key Scripts

- `train_phi4.py` — Training script. Run: `python train_phi4.py --L 128 --k 0.5 --l 0.022 --network ncsnpp --device cuda:0`
- `sample_phi4.py` — Sampling with EM/PC/MALA methods
- `sample_at_t0.py` — Sample evaluation at t=0
- `plot_loss_curves.py` — Loss decomposition plotting

## Import Pattern

All Python scripts use `sys.path.append("..")` to import root-level modules:
- `diffusion_lightning` (DiffusionModel)
- `networks` (ScoreNet, ScoreNetUNet, NCSNpp2D, etc.)
- `data` (FieldDataModule)

## Julia Scripts

- `Langevin.jl` — Langevin dynamics simulation (GPU-accelerated)
- `correlation_2D.jl` — Two-point correlation function measurement
- `CorrelationUtils.jl` — Shared correlation utilities
- `cumulant.jl` — Higher-order cumulant computation
- `Phi_4_model.jl` — phi^4 model definitions

## Subdirectories

- `acceptance_rate/` — MALA acceptance rate analysis: `calibrate_step_size.py`, `calibrate_t_mh.py`, `acceptance_vs_epoch.py`, `acceptance_scan.py`
- `score_quality_analysis/` — Score quality metrics by kappa and L
- `trainingdata/` — Julia-generated .jld2 training data (gitignored)
- `phi4_L*_k*_ncsnpp/` — Training output directories with `models/` and `data/` (gitignored)
