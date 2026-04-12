"""
Train Diffusion Model on CIFAR-10 Grayscale Images
Based on DMasSQ_lightning.py, adapted for natural images
"""

import os
import sys
sys.path.append("..")
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from diffusion_lightning import DiffusionModel, grab
from cifar10_datamodule import CIFAR10GrayDataModuleFast, CIFAR10_CLASSES

# Create directories
for folder in ['data', 'figures', 'models']:
    os.makedirs(folder, exist_ok=True)

# Check GPU
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    gpu_id = 0  # Change this if needed
else:
    print("No GPU available, using CPU")
    gpu_id = None

# ==================== Hyperparameters ====================
L = 32              # CIFAR-10 image size is 32x32
sigma = 25.0        # SDE noise scale
n_epochs = 10000      # Training epochs
batch_size = 128    # Batch size (can be larger for small images)
lr = 1e-4           # Learning rate (slightly lower for image data)
num_steps = 500     # Sampling steps

# Class filter: train on specific class(es)
# Options:
#   - None: use all classes
#   - 'cat': only cat images
#   - 'dog': only dog images
#   - ['cat', 'dog']: cat and dog images
#   - 3: class index for cat
# Available classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
class_filter = 'cat'  # Change this to train on different class(es)

# ==================== Data Module ====================
# normalize=False: keep data in original [0, 1] range (no conversion to [-1, 1])
data_module = CIFAR10GrayDataModuleFast(
    data_dir='./data/cifar10',
    batch_size=batch_size,
    normalize=False,  # Keep original [0, 1] range
    num_workers=4,
    class_filter=class_filter,
)
data_module.prepare_data()
data_module.setup()

print(f"Training samples: {len(data_module.train_data)}")

# Get class name for saving
if class_filter is not None:
    if isinstance(class_filter, list):
        class_name = '_'.join(str(c) for c in class_filter)
    else:
        class_name = str(class_filter)
else:
    class_name = 'all'

# ==================== Save Training Data ====================
# Save training data in format (Lx, Ly, N) for correlation analysis
print("Saving training data for correlation analysis...")
train_images_all = data_module.train_data.tensors[0]  # (N, 1, 32, 32)
train_images_np = train_images_all.numpy()  # Already in [0, 1] range (no renorm needed)
# Reshape from (N, 1, Lx, Ly) to (Lx, Ly, N)
train_cfgs = train_images_np[:, 0, :, :].transpose(1, 2, 0)  # (32, 32, N)
train_data_path = f'data/cifar10_{class_name}_train_{L}x{L}.npy'
np.save(train_data_path, train_cfgs)
print(f"Training data saved to: {train_data_path}")
print(f"  Shape: {train_cfgs.shape}, Range: [{train_cfgs.min():.3f}, {train_cfgs.max():.3f}]")

# ==================== Model ====================
# Note: periodic=False for natural images (no periodic boundary conditions)
# symmetry_aug=False since natural images don't have Z2/D4 symmetries
model = DiffusionModel(
    sigma=sigma,
    lr=lr,
    L=L,
    periodic=False,      # Natural images don't need periodic padding
    symmetry_aug=False,  # No physics symmetries for natural images
    z2_arch=False,
    d4_arch=False,
)

# ==================== Callbacks ====================
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='models/cifar10',
    filename=f'diffusion_cifar10_{class_name}' + '-{epoch:02d}-{train_loss_epoch:.4f}',
    save_top_k=3,
    monitor='train_loss_epoch',
    mode='min'
)

# Early stopping (optional)
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='train_loss_epoch',
    patience=20,
    mode='min',
    verbose=True,
)

# Learning rate monitor
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# ==================== Trainer ====================
trainer = pl.Trainer(
    max_epochs=n_epochs,
    accelerator='gpu' if gpu_id is not None else 'cpu',
    devices=[gpu_id] if gpu_id is not None else None,
    callbacks=[checkpoint_callback, lr_monitor],
    enable_progress_bar=True,
    log_every_n_steps=50,
)

# ==================== Training ====================
print("\n" + "="*50)
print("Starting Training...")
print("="*50 + "\n")

trainer.fit(model, data_module)

print("\n" + "="*50)
print("Training Complete!")
print(f"Best model saved to: {checkpoint_callback.best_model_path}")
print("="*50 + "\n")

# ==================== Generate Samples ====================
print("Generating samples...")

# Load best model
best_model = DiffusionModel.load_from_checkpoint(checkpoint_callback.best_model_path)
if gpu_id is not None:
    best_model = best_model.cuda(gpu_id)
best_model.eval()

# Number of samples to generate
num_samples_gen = 1000  # Generate more samples for correlation analysis

# Generate samples
with torch.no_grad():
    samples = best_model.sample(
        num_samples=num_samples_gen,
        num_steps=num_steps,
        eps=1e-3,
        sample_batch_size=128,
        show_progress=True,
    )

# Convert to numpy (no renorm needed since normalize=False)
samples_np = grab(samples)
# No renorm or clip - keep raw output values

# ==================== Save Generated Samples ====================
# Save in format (Lx, Ly, N) for correlation analysis
print("Saving generated samples for correlation analysis...")
# Reshape from (N, 1, Lx, Ly) to (Lx, Ly, N)
gen_cfgs = samples_np[:, 0, :, :].transpose(1, 2, 0)  # (32, 32, N)
gen_data_path = f'data/cifar10_{class_name}_dm_{L}x{L}.npy'
np.save(gen_data_path, gen_cfgs)
print(f"Generated samples saved to: {gen_data_path}")
print(f"  Shape: {gen_cfgs.shape}, Range: [{gen_cfgs.min():.3f}, {gen_cfgs.max():.3f}]")

# ==================== Visualization ====================
print("Saving visualizations...")

# Plot generated samples (first 64, raw output)
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples_np[i, 0], cmap='gray')  # No vmin/vmax to see actual range
    ax.axis('off')
plt.suptitle(f'Generated CIFAR-10 {class_name.upper()} Images (Raw)', fontsize=16)
plt.tight_layout()
plt.savefig(f'figures/cifar10_{class_name}_generated.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot comparison: real vs generated
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Get some real samples from training data (already in [0, 1] range)
real_batch = next(iter(data_module.train_dataloader()))
real_images = real_batch[0][:8].numpy()  # No renorm needed

# Top row: real images
for i in range(8):
    axes[0, i].imshow(real_images[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Real', fontsize=12, loc='left')

# Bottom row: generated images (raw output, may exceed [0,1])
for i in range(8):
    axes[1, i].imshow(samples_np[i, 0], cmap='gray')  # No vmin/vmax
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Generated', fontsize=12, loc='left')

plt.suptitle(f'Real vs Generated CIFAR-10 {class_name.upper()} Images', fontsize=14)
plt.tight_layout()
plt.savefig(f'figures/cifar10_{class_name}_real_vs_generated.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved to figures/")
print("Done!")
