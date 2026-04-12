"""
CIFAR-10 DataModule for Diffusion Model Training
Converts RGB images to grayscale for compatibility with existing ScoreNet architecture
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from typing import Union, List, Optional


# CIFAR-10 class names mapping
CIFAR10_CLASSES = {
    0: 'airplane',
    1: 'automobile', 
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Reverse mapping: name -> index
CIFAR10_CLASS_TO_IDX = {v: k for k, v in CIFAR10_CLASSES.items()}


def get_class_indices(class_filter: Union[int, str, List[Union[int, str]], None]) -> Optional[List[int]]:
    """
    Convert class filter to list of class indices.
    
    Args:
        class_filter: Can be:
            - None: use all classes
            - int: single class index (0-9)
            - str: single class name ('cat', 'dog', etc.)
            - list: multiple classes (ints or strings)
    
    Returns:
        List of class indices or None (for all classes)
    """
    if class_filter is None:
        return None
    
    if isinstance(class_filter, (int, str)):
        class_filter = [class_filter]
    
    indices = []
    for c in class_filter:
        if isinstance(c, int):
            if c < 0 or c > 9:
                raise ValueError(f"Class index must be 0-9, got {c}")
            indices.append(c)
        elif isinstance(c, str):
            c_lower = c.lower()
            if c_lower not in CIFAR10_CLASS_TO_IDX:
                raise ValueError(f"Unknown class name: {c}. Valid names: {list(CIFAR10_CLASS_TO_IDX.keys())}")
            indices.append(CIFAR10_CLASS_TO_IDX[c_lower])
        else:
            raise TypeError(f"class_filter must be int, str, or list, got {type(c)}")
    
    return indices


class CIFAR10GrayDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CIFAR-10 grayscale images (training only)."""

    def __init__(
        self,
        data_dir='./data/cifar10',
        batch_size=64,
        normalize=True,
        num_workers=4,
        class_filter: Union[int, str, List[Union[int, str]], None] = None,
    ):
        """
        Args:
            data_dir: Directory to store/load CIFAR-10 data
            batch_size: Batch size for training
            normalize: Whether to normalize to [-1, 1]
            num_workers: Number of data loading workers
            class_filter: Filter by class. Can be:
                - None: use all classes (default)
                - int: single class index (0-9)
                - str: single class name ('cat', 'dog', etc.)
                - list: multiple classes
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.class_indices = get_class_indices(class_filter)

        self.data_mean = None
        self.data_std = None

    def prepare_data(self):
        """Download CIFAR-10 training dataset."""
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        """Setup training dataset."""
        # Define transforms: convert to grayscale and tensor
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])

        # Load training dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=transform,
            download=False
        )

        # Convert to tensors for faster loading
        train_images = []
        train_labels = []
        for img, label in train_dataset:
            # Filter by class if specified
            if self.class_indices is None or label in self.class_indices:
                train_images.append(img)
                train_labels.append(label)

        train_images = torch.stack(train_images)  # (N, 1, 32, 32)
        train_labels = torch.tensor(train_labels)

        # Normalize to [-1, 1] range (common for diffusion models)
        if self.normalize:
            train_images = train_images * 2 - 1
            self.data_mean = 0.0
            self.data_std = 1.0

        # Print info
        if self.class_indices is not None:
            class_names = [CIFAR10_CLASSES[i] for i in self.class_indices]
            print(f"Filtered classes: {class_names} (indices: {self.class_indices})")
        print(f"Train images shape: {train_images.shape}")
        print(f"Train images range: [{train_images.min():.3f}, {train_images.max():.3f}]")

        # Create TensorDataset
        self.train_data = TensorDataset(train_images, train_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def renorm(self, y):
        """Renormalize data back to [0, 1] scale for visualization."""
        if self.normalize:
            return (y + 1) / 2
        return y


class CIFAR10GrayDataModuleFast(pl.LightningDataModule):
    """
    Faster version using pre-loaded numpy arrays (training only).
    Good for repeated experiments.
    """

    def __init__(
        self,
        data_dir='./data/cifar10',
        batch_size=64,
        normalize=True,
        num_workers=4,
        class_filter: Union[int, str, List[Union[int, str]], None] = None,
    ):
        """
        Args:
            data_dir: Directory to store/load CIFAR-10 data
            batch_size: Batch size for training
            normalize: Whether to normalize to [-1, 1]
            num_workers: Number of data loading workers
            class_filter: Filter by class. Can be:
                - None: use all classes (default)
                - int: single class index (0-9), e.g., 3 for cat
                - str: single class name, e.g., 'cat', 'dog'
                - list: multiple classes, e.g., ['cat', 'dog'] or [3, 5]
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.class_indices = get_class_indices(class_filter)

    def prepare_data(self):
        """Download CIFAR-10 training dataset."""
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)

    def _rgb_to_gray(self, rgb_images):
        """
        Convert RGB images to grayscale using luminosity method.
        Input: (N, H, W, 3) uint8
        Output: (N, 1, H, W) float32
        """
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        gray = np.sum(rgb_images.astype(np.float32) * weights, axis=-1, keepdims=True)
        gray = gray / 255.0
        gray = np.transpose(gray, (0, 3, 1, 2))
        return gray

    def setup(self, stage=None):
        """Setup training dataset."""
        # Load raw CIFAR-10 training data
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=False
        )

        # Get numpy arrays directly
        train_images = train_dataset.data  # (50000, 32, 32, 3) uint8
        train_labels = np.array(train_dataset.targets)

        # Filter by class if specified
        if self.class_indices is not None:
            mask = np.isin(train_labels, self.class_indices)
            train_images = train_images[mask]
            train_labels = train_labels[mask]

        # Convert to grayscale
        train_gray = self._rgb_to_gray(train_images)

        # Normalize to [-1, 1]
        if self.normalize:
            train_gray = train_gray * 2 - 1

        # Convert to tensors
        train_images_t = torch.from_numpy(train_gray).float()
        train_labels_t = torch.from_numpy(train_labels).long()

        # Print info
        if self.class_indices is not None:
            class_names = [CIFAR10_CLASSES[i] for i in self.class_indices]
            print(f"Filtered classes: {class_names} (indices: {self.class_indices})")
        print(f"Train images shape: {train_images_t.shape}")
        print(f"Train images range: [{train_images_t.min():.3f}, {train_images_t.max():.3f}]")

        self.train_data = TensorDataset(train_images_t, train_labels_t)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def renorm(self, y):
        """Renormalize data back to [0, 1] scale for visualization."""
        if self.normalize:
            return (y + 1) / 2
        return y


if __name__ == "__main__":
    print("CIFAR-10 Classes:")
    for idx, name in CIFAR10_CLASSES.items():
        print(f"  {idx}: {name}")
    print()

    # Test with single class filter (cat)
    print("=" * 50)
    print("Testing with class_filter='cat'")
    print("=" * 50)
    dm = CIFAR10GrayDataModuleFast(batch_size=32, class_filter='cat')
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")

    # Test with multiple classes
    print("\n" + "=" * 50)
    print("Testing with class_filter=['cat', 'dog']")
    print("=" * 50)
    dm2 = CIFAR10GrayDataModuleFast(batch_size=32, class_filter=['cat', 'dog'])
    dm2.prepare_data()
    dm2.setup()

    batch2 = next(iter(dm2.train_dataloader()))
    images2, labels2 = batch2
    print(f"Batch shape: {images2.shape}")
    print(f"Unique labels in batch: {torch.unique(labels2).tolist()}")

    # Visualize
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        img = dm.renorm(images[i, 0]).numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{CIFAR10_CLASSES[labels[i].item()]}")
        ax.axis('off')
    plt.suptitle("CIFAR-10 Cat Images (Grayscale)", fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/cifar10_cat_samples.png', dpi=150)
    plt.show()
    print("Saved cat samples to figures/cifar10_cat_samples.png")
