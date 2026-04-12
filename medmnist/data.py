"""
Data modules for MedMNIST datasets.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from medmnist import ChestMNIST


class ChestMNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ChestMNIST.
    
    Args:
        batch_size: Batch size for training.
        size: Image size (default 64).
        device: Device to store data on (e.g., "cuda:0"). If None, uses CPU with DataLoader.
    """

    def __init__(self, batch_size=64, size=64, device=None):
        super().__init__()
        self.batch_size = batch_size
        self.size = size
        self.device = device
        self.data_on_gpu = None

    def setup(self, stage=None):
        # Load dataset
        train_dataset = ChestMNIST(split="train", size=self.size)
        
        # Convert to tensor: ChestMNIST returns PIL images, need to convert
        images = []
        labels = []
        for i in range(len(train_dataset)):
            img, label = train_dataset[i]
            # Convert PIL to numpy, then to tensor, normalize to [0, 1]
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            # img shape is (H, W) or (H, W, C), need (C, H, W)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)
            elif img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
            images.append(img_tensor)
            labels.append(torch.tensor(label))
        
        images = torch.stack(images)  # (N, C, H, W)
        labels = torch.stack(labels)
        
        print(f"ChestMNIST loaded: {images.shape}, range [{images.min():.3f}, {images.max():.3f}]")
        
        if self.device:
            # Move entire dataset to GPU
            self.data_on_gpu = images.to(self.device)
            self.labels_on_gpu = labels.to(self.device)
        else:
            self.images = images
            self.labels = labels

    def train_dataloader(self):
        if self.device:
            return GPUDataLoader(self.data_on_gpu, self.labels_on_gpu, self.batch_size)
        else:
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(self.images, self.labels)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


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
