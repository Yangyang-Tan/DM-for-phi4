"""
Data modules for diffusion models.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


class MNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MNIST.
    
    Args:
        data_dir: Directory to store MNIST data.
        batch_size: Batch size for training.
        device: Device to store data on (e.g., "cuda:0"). If None, uses CPU with DataLoader.
    """

    def __init__(self, data_dir="./data", batch_size=64, device=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.data_on_gpu = None

    def prepare_data(self):
        torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        # Load entire dataset
        dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, transform=transforms.ToTensor(), download=False
        )
        
        # Convert to single tensor
        images = torch.stack([dataset[i][0] for i in range(len(dataset))])  # (60000, 1, 28, 28)
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        if self.device:
            # Move entire dataset to GPU
            self.data_on_gpu = images.to(self.device)
            self.labels_on_gpu = labels.to(self.device)
        else:
            self.train_data = TensorDataset(images, labels)

    def train_dataloader(self):
        if self.device:
            # Return a simple generator that yields batches from GPU
            return GPUDataLoader(self.data_on_gpu, self.labels_on_gpu, self.batch_size)
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)


class GPUDataLoader:
    """Simple DataLoader that keeps data on GPU and yields random batches."""
    
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.n_samples = len(data)
    
    def __iter__(self):
        # Random permutation each epoch
        indices = torch.randperm(self.n_samples, device=self.data.device)
        for start in range(0, self.n_samples, self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield self.data[idx], self.labels[idx]
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class FieldDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for field configurations.

    If `device` is given, the entire dataset is pinned on that GPU at setup()
    and the train_dataloader yields batches by slicing GPU tensors — no
    per-epoch host→device transfer.
    """

    def __init__(self, data_path, batch_size=64, normalize=True, device=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device

        self.cfgs_min = None
        self.cfgs_max = None
        self.data_on_gpu = None

    def setup(self, stage=None):
        import h5py
        with h5py.File(self.data_path, "r") as f:
            cfgs = np.array(f["cfgs"])

        # Normalize if needed
        if self.normalize:
            self.cfgs_min = cfgs.min()
            self.cfgs_max = cfgs.max()
            cfgs = ((cfgs - self.cfgs_min) / (self.cfgs_max - self.cfgs_min) - 0.5) * 2

        # Convert to tensors
        configs = torch.from_numpy(cfgs).unsqueeze(1).float()

        if self.device:
            self.data_on_gpu = configs.to(self.device)
            size_gb = self.data_on_gpu.nbytes / 1e9
            print(f"Field data loaded to {self.device} ({size_gb:.2f} GB, shape={tuple(self.data_on_gpu.shape)})")

        # Training data (no separate test/validation split here)
        self.train_data = TensorDataset(configs, configs)

    def train_dataloader(self):
        if self.device:
            return GPUDataLoader(self.data_on_gpu, self.data_on_gpu, self.batch_size)
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def renorm(self, y):
        """Renormalize data back to original scale."""
        if self.cfgs_min is None:
            return y
        return (y / 2 + 0.5) * (self.cfgs_max - self.cfgs_min) + self.cfgs_min
