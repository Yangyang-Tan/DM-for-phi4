import os
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from diffusion_lightning_3d_full import (
    DiffusionModel3D, FieldDataModule3D, 
    grab, get_mag_3d, get_abs_mag_3d, get_chi2_3d, get_UL_3d, mag_3d
)

# Create directories
for folder in ['data', 'figures', 'models']:
    os.makedirs(folder, exist_ok=True)
# Lattice parameters
L = 32          # 3D lattice size: L x L x L (use smaller L for 3D due to memory)
k = 0.5        # Hopping parameter
l = 2.5       # Coupling
sigma = 25.0    # SDE noise scale

# Training parameters
n_epochs = 1500
batch_size = 64  # Smaller batch size for 3D due to memory constraints
lr = 1e-3

# Data path (adjust to your 3D data)
data_path = f"data/cfgs_k={k}_l={l}_32^3_t=10.jld2"

# Create the 3D diffusion model with MEMORY OPTIMIZATION
model = DiffusionModel3D(
    sigma=sigma,
    lr=lr,
    L=L,
    periodic=True,           # Use periodic boundary conditions
    symmetry_aug=False,       # Apply symmetry augmentations during training
    z2_arch=False,           # Z2 symmetric architecture (optional)
    cubic_arch=False,        # Cubic equivariant architecture (expensive: 24x compute)
    full_cubic_group=False,  # Use full Oh group (48 elements) vs O group (24)
    # ===== MEMORY OPTIMIZATION OPTIONS =====
    use_checkpoint=False,     # Enable gradient checkpointing (~30-50% memory savings)
)

# Print model summary
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Gradient checkpointing: {model.use_checkpoint}")

# Set up data module (uncomment when you have data)
data_module = FieldDataModule3D(
    data_path=data_path,
    batch_size=batch_size,
    normalize=True
)
data_module.setup()
print(f"Training samples: {len(data_module.train_data)}")

# Set up training
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='models/3D_2',
    filename=f'diffusion_L{L}_k{k}_l{l}_32^3_t=10',
    save_top_k=3,
    monitor='train_loss_epoch',  # Monitor epoch-averaged loss
    mode='min'
)

checkpoint_callback2=pl.callbacks.ModelCheckpoint(
    dirpath='models/3D_2',
    filename=f'diffusion_L{L}_k{k}_l{l}_32^3_t=10-{{epoch:02d}}',
    save_top_k=-1,
    every_n_epochs=20,
)

# ===== MEMORY OPTIMIZATION IN TRAINER =====
# 1. precision="16-mixed" or "bf16-mixed": Mixed precision training (~50% memory savings)
# 2. accumulate_grad_batches: Simulate larger batches without more memory
# 3. gradient_clip_val: Prevent gradient explosions with AMP

trainer = pl.Trainer(
    max_epochs=n_epochs,
    accelerator='gpu',
    devices=[2],
    callbacks=[checkpoint_callback, checkpoint_callback2],
    enable_progress_bar=True,
    # ===== MEMORY OPTIMIZATION OPTIONS =====
    precision="16-mixed",        # Use AMP (float16) - ~50% memory savings!
    accumulate_grad_batches=1,   # Effective batch = batch_size * 4 = 128
    # gradient_clip_val=1.0,       # Recommended with AMP to prevent gradient explosions
)

print(f"Training with precision: {trainer.precision}")
print(f"Gradient accumulation: {trainer.accumulate_grad_batches} steps")
print(f"Effective batch size: {batch_size * trainer.accumulate_grad_batches}")

# Train the model
# trainer.fit(model, data_module)
trainer.fit(model, data_module, ckpt_path="models/3D_2/diffusion_L32_k0.5_l2.5_32^3_t=10-epoch=499.ckpt")
# trainer.save_checkpoint(f"models/diffusion_L{L}_k{k}_l{l}_32^3-last.ckpt")