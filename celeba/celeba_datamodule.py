"""
CelebA DataModule for Diffusion Model Training.
Loads preprocessed grayscale .npy cache (created by preprocess_celeba.py).

Available sizes: 64x64, 128x128 (~200k images each).
"""

import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


class GPUDataLoader:
    """Simple DataLoader that keeps data on GPU and yields random batches."""

    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.n_samples = len(data)

    def __iter__(self):
        indices = torch.randperm(self.n_samples, device=self.data.device)
        for start in range(0, self.n_samples, self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield self.data[idx], self.labels[idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class CelebAGrayDataModule(pl.LightningDataModule):
    """
    CelebA grayscale DataModule. Loads preprocessed .npy cache.

    Run preprocess_celeba.py first to create the cache.

    Args:
        data_dir: Directory containing preprocessed .npy files.
        image_size: 64 or 128.
        batch_size: Batch size for training.
        normalize: Whether to normalize to [-1, 1].
        num_workers: Number of data loading workers (ignored if device is set).
        max_samples: Limit number of samples (None = use all ~200k).
        device: GPU device string (e.g. 'cuda:0') to load all data onto GPU.
                None = use CPU DataLoader with num_workers.
    """

    def __init__(
        self,
        data_dir='./data',
        image_size=64,
        batch_size=64,
        normalize=True,
        num_workers=4,
        max_samples=None,
        device=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.device = device

    def prepare_data(self):
        path = os.path.join(self.data_dir, f'celeba_gray{self.image_size}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preprocessed file not found: {path}\n"
                "Run 'python preprocess_celeba.py' first.")

    def setup(self, stage=None):
        path = os.path.join(self.data_dir, f'celeba_gray{self.image_size}.npy')
        images = np.load(path)  # (N, 1, H, W) float32 [0, 1]

        if self.max_samples is not None:
            images = images[:self.max_samples]

        if self.normalize:
            images = images * 2 - 1

        images_t = torch.from_numpy(images).float()
        labels_t = torch.zeros(len(images_t), dtype=torch.long)

        print(f"CelebA {self.image_size}x{self.image_size}: {images_t.shape}")
        print(f"Range: [{images_t.min():.3f}, {images_t.max():.3f}]")

        if self.device:
            self.data_on_gpu = images_t.to(self.device)
            self.labels_on_gpu = labels_t.to(self.device)
            size_gb = self.data_on_gpu.nbytes / 1e9
            print(f"Data loaded to {self.device} ({size_gb:.1f} GB)")

        self.train_data = TensorDataset(images_t, labels_t)

    def train_dataloader(self):
        if self.device:
            return GPUDataLoader(self.data_on_gpu, self.labels_on_gpu, self.batch_size)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def renorm(self, y):
        if self.normalize:
            return (y + 1) / 2
        return y
