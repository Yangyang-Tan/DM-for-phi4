"""
Data modules for 3D diffusion models on lattice field theory.

Loads all data into memory for fast training.
Saves normalization parameters to cache file to avoid recomputing each time.
"""

import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import h5py


class GPUDataLoader3D:
    """Yields random batches from a GPU-resident tensor — zero host→device transfer."""
    def __init__(self, data, labels, batch_size):
        self.data = data; self.labels = labels; self.batch_size = batch_size
        self.n = len(data)
    def __iter__(self):
        idx = torch.randperm(self.n, device=self.data.device)
        for s in range(0, self.n, self.batch_size):
            sel = idx[s:s+self.batch_size]
            yield self.data[sel], self.labels[sel]
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


class FieldDataModule3D(pl.LightningDataModule):
    """PyTorch Lightning DataModule for 3D field configurations.

    If `device` is given, the entire dataset is pinned on that GPU at setup()
    and the train_dataloader yields batches by slicing GPU tensors — no
    per-epoch host→device transfer.
    """

    def __init__(self, data_path, batch_size=64, normalize=True, num_workers=4, device=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.device = device

        self.cfgs_min = None
        self.cfgs_max = None
        self.data_on_gpu = None

        # Path to save normalization parameters (next to data file)
        self.norm_cache_path = data_path + ".norm.json"

    def _load_norm_cache(self):
        """Load cached normalization parameters if available."""
        if os.path.exists(self.norm_cache_path):
            with open(self.norm_cache_path, "r") as f:
                cache = json.load(f)
            self.cfgs_min = cache["cfgs_min"]
            self.cfgs_max = cache["cfgs_max"]
            print(f"Loaded normalization from cache: [{self.cfgs_min:.4f}, {self.cfgs_max:.4f}]")
            return True
        return False

    def _save_norm_cache(self):
        """Save normalization parameters to cache file."""
        cache = {"cfgs_min": float(self.cfgs_min), "cfgs_max": float(self.cfgs_max)}
        with open(self.norm_cache_path, "w") as f:
            json.dump(cache, f)
        print(f"Saved normalization to cache: {self.norm_cache_path}")

    def setup(self, stage=None):
        print(f"Loading data from: {self.data_path}")
        
        with h5py.File(self.data_path, "r") as f:
            cfgs = np.array(f["cfgs"])
        
        print(f"Raw data shape: {cfgs.shape}")
        
        # Expect cfgs to be 4D: (num_samples, L, L, L) for 3D lattice
        # or 4D: (L, L, L, num_samples) which needs transposing
        if cfgs.ndim == 4:
            # Check if last dimension is larger (likely num_samples)
            if cfgs.shape[-1] > cfgs.shape[0]:
                # Shape is (L, L, L, num_samples), transpose to (num_samples, L, L, L)
                print("Transposing data: (L,L,L,N) -> (N,L,L,L)")
                cfgs = cfgs.transpose(3, 0, 1, 2)
        elif cfgs.ndim == 3:
            # Single sample or need to add batch dimension
            cfgs = cfgs[np.newaxis, ...]
        
        print(f"Data shape after processing: {cfgs.shape}")
        print(f"Number of samples: {cfgs.shape[0]}")

        # Try to load cached normalization, otherwise compute and save
        if self.normalize:
            if not self._load_norm_cache():
                print("Computing normalization parameters (first time only)...")
                self.cfgs_min = float(cfgs.min())
                self.cfgs_max = float(cfgs.max())
                print(f"Normalization range: [{self.cfgs_min:.4f}, {self.cfgs_max:.4f}]")
                self._save_norm_cache()
            
            # Normalize to [-1, 1]
            cfgs = ((cfgs - self.cfgs_min) / (self.cfgs_max - self.cfgs_min) - 0.5) * 2

        # Convert to tensors with channel dimension
        # Shape: (num_samples, 1, L, L, L)
        configs = torch.from_numpy(cfgs).unsqueeze(1).float()
        
        # Free numpy array memory
        del cfgs
        
        print(f"Tensor shape: {configs.shape}")
        print(f"Memory usage: {configs.element_size() * configs.nelement() / 1e9:.2f} GB")

        if self.device:
            self.data_on_gpu = configs.to(self.device)
            print(f"Data loaded to {self.device} ({self.data_on_gpu.nbytes/1e9:.2f} GB)")

        # Training data
        self.train_data = TensorDataset(configs, configs)

    def train_dataloader(self):
        if self.device:
            return GPUDataLoader3D(self.data_on_gpu, self.data_on_gpu, self.batch_size)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def renorm(self, y):
        """Renormalize data back to original scale."""
        if self.cfgs_min is None:
            return y
        return (y / 2 + 0.5) * (self.cfgs_max - self.cfgs_min) + self.cfgs_min
