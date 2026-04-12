# 3D phi^4 Lattice Field Theory

3D scalar phi^4 theory diffusion model. Lattice size L=64; hopping parameter kappa=0.18,0.1923,0.2; coupling lambda=0.9.

## Local Modules (in this directory, NOT root)

- `diffusion_lightning_3d.py` — `DiffusionModel3D` Lightning module (mirrors root 2D version with Conv3d)
- `networks_3d.py` — 3D score networks: `ScoreNet3D`, `ScoreNetUNet3D`, `NCSNpp3D`
- `data_3d.py` — `FieldDataModule3D` with norm caching to `.norm.json`

## Key Scripts

- `train_phi4.py` — Training with `torch.compile` + bf16-mixed precision. Requires GPU ≥24GB (4090/5090). Run: `python train_phi4.py --L 64 --k 0.2 --l 0.9 --network ncsnpp --device cuda:0`
- `sample_phi4.py` — Sampling script
- `compute_fid_3d.py` — FID computation for 3D samples
- `plot_loss_curves_3d.py` — Loss decomposition plotting

## Import Pattern

Uses `sys.path.append("..")` to import root-level `networks` and `data`. Local 3D modules (`diffusion_lightning_3d`, `networks_3d`, `data_3d`) are imported directly.

## MALA Calibration Scripts

- `calibrate_step_size.py` — Step size calibration for MALA
- `calibrate_t_mh.py` — Metropolis-Hastings transition time calibration
- `acceptance_vs_epoch.py` — Acceptance rate vs training epoch analysis

## Subdirectories

- `score_quality_analysis/` — Score quality metrics for 3D models
- `trainingdata/` — Julia-generated .jld2 training data (gitignored)
- `phi4_3d_L*_k*_ncsnpp/` — Training output directories (gitignored)
