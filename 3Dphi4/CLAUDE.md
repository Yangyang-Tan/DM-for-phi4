# 3D phi^4 Lattice Field Theory

3D scalar phi^4 theory diffusion model. Lattice size L=64; hopping parameter kappa=0.18,0.1923,0.2; coupling lambda=0.9.

## Local Modules (in this directory, NOT root)

- `diffusion_lightning_3d.py` — `DiffusionModel3D` Lightning module (mirrors root 2D version with Conv3d)
- `networks_3d.py` — 3D score networks: `ScoreNet3D`, `ScoreNetUNet3D`, `NCSNpp3D`
- `data_3d.py` — `FieldDataModule3D` with norm caching to `.norm.json`

## Key Scripts

- `train_phi4.py` — Training with `torch.compile` + bf16-mixed precision. Requires GPU ≥24GB (4090/5090). Run: `python train_phi4.py --L 64 --k 0.2 --l 0.9 --network ncsnpp --device cuda:0`
- `sample_phi4.py` / `sample_phi4_3d_sweep.py` — Sampling (single + multi-ckpt sweep)
- `analysis/compute_fid_3d.py` — FID computation for 3D samples
- `analysis/plot_loss_curves_3d.py` — Loss decomposition plotting
- All other data-analysis scripts live under `analysis/`

## Import Pattern

Top-level scripts (`train_phi4.py`, `sample_phi4*.py`) use `sys.path.append("..")` to import root-level `networks`/`data`. Scripts under `analysis/` use `Path(__file__).resolve().parents[1]` for `3Dphi4/` and `parents[2]` for repo root. Local 3D modules (`diffusion_lightning_3d`, `networks_3d`, `data_3d`) live at `3Dphi4/`.

## MALA Calibration Scripts (under `analysis/`)

- `analysis/calibrate_step_size.py` — Step size calibration for MALA
- `analysis/calibrate_t_mh.py` — Metropolis-Hastings transition time calibration
- `analysis/acceptance_vs_epoch.py` — Acceptance rate vs training epoch analysis

Run from `3Dphi4/`: `python analysis/calibrate_step_size.py --device cuda:2 --L 64 --k 0.2`

## Subdirectories

- `analysis/` — Python data-analysis scripts (FID, loss curves, MALA calibration, epoch sweeps, EM-vs-DPM comparisons, etc.)
- `score_quality_analysis/` — Score quality metrics for 3D models
- `julia/` — Julia analysis/simulator scripts (`Langevin3D.jl`, `correlation*.jl`)
- `mathematica/` — `*.wls` plotting (`CumulantVsEpochPhi4L64_sigma2760.wls`)
- `scripts/` — Shell drivers (`run_sweep_L32.sh`, `run_sampling_all_epochs.sh`). Invoke from `3Dphi4/` via `bash scripts/<name>.sh`.
- `docs/` — Network architecture LaTeX diagrams (`ncsnpp3d_*.tex`)
- `results/` — All run outputs (mostly gitignored):
  - `calibration/` — acceptance/calibration figures and CSVs
  - `L{32,64}_k*_sigma*_logs/`, `stress_logs/` — sweep / stress-test logs
  - `sigma_comparison_3D/` — multi-epoch propagator caches and diagnostic figures (committed)
  - `sampling_all_epochs.log`
- `trainingdata/` — Julia-generated .jld2 training data (gitignored)
- `runs/` — All training output directories (gitignored). Layout: `runs/phi4_3d_L{L}_k{k}_l{λ}_ncsnpp[_sigma{σ}]/{models,data}/`. Train/sample/analysis scripts hardcode the `runs/` prefix.
