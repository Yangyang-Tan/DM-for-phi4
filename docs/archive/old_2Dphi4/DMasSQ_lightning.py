import os
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from diffusion_lightning import (
    DiffusionModel, FieldDataModule, 
    grab, get_mag, get_abs_mag, get_chi2, get_UL, mag
)

# Create directories
for folder in ['data', 'figures', 'models']:
    os.makedirs(folder, exist_ok=True)

print(f"Using GPU: {torch.cuda.get_device_name(2)}")
L = 128          
k = 0.5  
l = 0.022       
sigma = 25.0


n_epochs = 50
batch_size = 64
lr = 1e-3

num_steps = 1000
# num_samples = 1024

data_path = f"data/cfgs_k={k}_l={l}_128^2_t=10.jld2"
data_module = FieldDataModule(
    data_path=data_path,
    batch_size=batch_size,
    normalize=True
)
data_module.setup()

print(f"Training samples: {len(data_module.train_data)}")
model = DiffusionModel(sigma=sigma, lr=lr, L=L,periodic=True,symmetry_aug=True)


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='models/2D-10',
    filename=f'diffusion_L{L}_k{k}_l{l}_128^2_t=10',
    save_top_k=3,
    monitor='train_loss_epoch',  # Monitor epoch-averaged loss
    mode='min'
)

checkpoint_callback2=pl.callbacks.ModelCheckpoint(
    dirpath='models/2D-10',
    filename=f'diffusion_L{L}_k{k}_l{l}_128^2_t=10-{{epoch:02d}}-{{step:04d}}',
    save_top_k=-1,
    every_n_train_steps=5,
)


trainer = pl.Trainer(
    max_epochs=n_epochs,
    accelerator='gpu',
    devices=[1],
    callbacks=[checkpoint_callback, checkpoint_callback2],
    enable_progress_bar=True,
)

trainer.fit(model, data_module)
