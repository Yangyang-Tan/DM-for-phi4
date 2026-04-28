"""
Data modules for diffusion models.
"""

import os
import json

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


def _normalize_pm1(arr, lo, hi):
    return ((arr - lo) / (hi - lo) - 0.5) * 2


def _renorm_pm1(y, lo, hi):
    return (y / 2 + 0.5) * (hi - lo) + lo


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
    """PyTorch Lightning DataModule for 2D / 3D lattice field configurations.

    Reads HDF5 with key ``cfgs`` of shape ``(N, L, L)`` for 2D or
    ``(N, L, L, L)`` for 3D. Auto-detects ``(L, L, L, N)`` 3D layouts and
    transposes them.

    If `device` is given, the entire dataset is pinned on that GPU at setup()
    and the train_dataloader yields batches by slicing GPU tensors — no
    per-epoch host→device transfer.

    If `cache_norm` is True, normalisation min/max are cached to
    ``<data_path>.norm.json`` on first use and reloaded on subsequent runs.
    """

    def __init__(self, data_path, batch_size=64, normalize=True, device=None,
                 num_workers=0, cache_norm=False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.num_workers = num_workers
        self.cache_norm = cache_norm
        self.norm_cache_path = data_path + ".norm.json" if cache_norm else None

        self.cfgs_min = None
        self.cfgs_max = None
        self.data_on_gpu = None

    def _load_norm_cache(self):
        if not (self.cache_norm and os.path.exists(self.norm_cache_path)):
            return False
        with open(self.norm_cache_path, "r") as f:
            cache = json.load(f)
        self.cfgs_min = cache["cfgs_min"]
        self.cfgs_max = cache["cfgs_max"]
        print(f"Loaded normalization from cache: [{self.cfgs_min:.4f}, {self.cfgs_max:.4f}]")
        return True

    def _save_norm_cache(self):
        if not self.cache_norm:
            return
        cache = {"cfgs_min": float(self.cfgs_min), "cfgs_max": float(self.cfgs_max)}
        with open(self.norm_cache_path, "w") as f:
            json.dump(cache, f)
        print(f"Saved normalization to cache: {self.norm_cache_path}")

    def setup(self, stage=None):
        import h5py
        with h5py.File(self.data_path, "r") as f:
            cfgs = np.array(f["cfgs"])

        # 3D layout autodetect: (L,L,L,N) -> (N,L,L,L)
        if cfgs.ndim == 4 and cfgs.shape[-1] > cfgs.shape[0]:
            cfgs = cfgs.transpose(3, 0, 1, 2)

        # Normalize if needed
        if self.normalize:
            if not self._load_norm_cache():
                self.cfgs_min = float(cfgs.min())
                self.cfgs_max = float(cfgs.max())
                self._save_norm_cache()
            cfgs = _normalize_pm1(cfgs, self.cfgs_min, self.cfgs_max)

        # Convert to tensors
        configs = torch.from_numpy(cfgs).unsqueeze(1).float()
        del cfgs

        if self.device:
            self.data_on_gpu = configs.to(self.device)
            size_gb = self.data_on_gpu.nbytes / 1e9
            print(f"Field data loaded to {self.device} ({size_gb:.2f} GB, shape={tuple(self.data_on_gpu.shape)})")

        # Training data (no separate test/validation split here)
        self.train_data = TensorDataset(configs, configs)

    def train_dataloader(self):
        if self.device:
            return GPUDataLoader(self.data_on_gpu, self.data_on_gpu, self.batch_size)
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def renorm(self, y):
        """Renormalize data back to original scale."""
        if self.cfgs_min is None:
            return y
        return _renorm_pm1(y, self.cfgs_min, self.cfgs_max)


class MultiLFieldDataModule(pl.LightningDataModule):
    """Multi-lattice-size DataModule for cross-L score-based training.

    Loads several jld2 files (one per L), pools them, and yields random batches
    where each batch contains a single L (since shapes must agree within a
    batch). Normalisation uses a single global (min, max) computed over the
    pooled data — the std is approximately L-independent for this physics so
    a uniform [-1,1] map is appropriate, and using one norm makes the
    network's input scale L-invariant.

    Per-step L is sampled with probability proportional to dataset size
    (i.e. one epoch ≈ one pass over all configurations).
    """

    def __init__(self, data_paths, batch_size=64, normalize=True, device=None):
        super().__init__()
        self.data_paths = list(data_paths)
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device

        self.cfgs_min = None
        self.cfgs_max = None
        self.data_by_L = None      # dict: L -> tensor (N, 1, L, L)
        self.Ls = None             # sorted list of L's available
        self.N_total = 0

    def setup(self, stage=None):
        import h5py
        self.N_total = 0
        raws = {}
        for path in self.data_paths:
            with h5py.File(path, "r") as f:
                cfgs = np.array(f["cfgs"])
            # Heuristic: detect sample axis (largest) and put it first
            sample_axis = int(np.argmax(cfgs.shape))
            if cfgs.ndim == 3 and sample_axis != 0:
                cfgs = np.moveaxis(cfgs, sample_axis, 0)
            L = int(cfgs.shape[1])
            assert cfgs.shape[2] == L, f"non-square lattice in {path}: {cfgs.shape}"
            raws[L] = cfgs.astype(np.float32)

        # Pooled global normalisation
        if self.normalize:
            mins = [r.min() for r in raws.values()]
            maxs = [r.max() for r in raws.values()]
            self.cfgs_min = float(min(mins))
            self.cfgs_max = float(max(maxs))
            print(f"Multi-L pooled norm: [{self.cfgs_min:.4f}, {self.cfgs_max:.4f}] "
                  f"over Ls = {sorted(raws.keys())}")
            for L in raws:
                raws[L] = _normalize_pm1(raws[L], self.cfgs_min, self.cfgs_max)

        self.data_by_L = {}
        for L, arr in raws.items():
            t = torch.from_numpy(arr).unsqueeze(1).float()
            if self.device:
                t = t.to(self.device)
            self.data_by_L[L] = t
            self.N_total += t.shape[0]
        self.Ls = sorted(self.data_by_L.keys())

        sizes = ", ".join(f"L={L}: N={self.data_by_L[L].shape[0]}" for L in self.Ls)
        if self.device:
            total_gb = sum(t.nbytes for t in self.data_by_L.values()) / 1e9
            print(f"Multi-L data on {self.device}: {sizes}  ({total_gb:.2f} GB)")
        else:
            print(f"Multi-L data on CPU: {sizes}")

    def train_dataloader(self):
        return MultiLBatchSampler(self.data_by_L, self.batch_size)

    def renorm(self, y):
        if self.cfgs_min is None:
            return y
        return _renorm_pm1(y, self.cfgs_min, self.cfgs_max)


class MultiLBatchSampler:
    """Iterator that yields one batch per step from a randomly-chosen L.

    `len(self)` corresponds to one pass over all configurations (across L's).
    L is sampled per step with probability proportional to dataset size.
    """

    def __init__(self, data_by_L, batch_size):
        self.data_by_L = data_by_L
        self.batch_size = batch_size
        self.Ls = sorted(data_by_L.keys())
        self.sizes = np.array([data_by_L[L].shape[0] for L in self.Ls], dtype=np.float64)
        self.probs = self.sizes / self.sizes.sum()
        self.n_steps = int(self.sizes.sum() // batch_size)

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.n_steps):
            L = self.Ls[int(rng.choice(len(self.Ls), p=self.probs))]
            data = self.data_by_L[L]
            idx = torch.randint(0, data.shape[0], (self.batch_size,), device=data.device)
            x = data[idx]
            yield x, x

    def __len__(self):
        return self.n_steps
