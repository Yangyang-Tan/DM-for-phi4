"""
STL-10 DataModule for Diffusion Model Training.
Loads preprocessed grayscale 64x64 .npy cache (created by preprocess_stl10.py).

STL-10 splits:
  - 'unlabeled': 100,000 images (no labels, best for unsupervised generation)
  - 'train':       5,000 labeled images (10 classes)
  - 'test':        8,000 labeled images (10 classes)

Classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
"""

import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Union, List, Optional


STL10_CLASSES = {
    0: 'airplane',
    1: 'bird',
    2: 'car',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'horse',
    7: 'monkey',
    8: 'ship',
    9: 'truck'
}

STL10_CLASS_TO_IDX = {v: k for k, v in STL10_CLASSES.items()}


def get_class_indices(class_filter: Union[int, str, List[Union[int, str]], None]) -> Optional[List[int]]:
    """Convert class filter to list of class indices."""
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
            if c_lower not in STL10_CLASS_TO_IDX:
                raise ValueError(f"Unknown class: {c}. Valid: {list(STL10_CLASS_TO_IDX.keys())}")
            indices.append(STL10_CLASS_TO_IDX[c_lower])
        else:
            raise TypeError(f"class_filter must be int, str, or list, got {type(c)}")
    return indices


class STL10GrayDataModule(pl.LightningDataModule):
    """
    STL-10 grayscale DataModule. Loads preprocessed .npy cache files.

    Run preprocess_stl10.py first to create the cache:
        python preprocess_stl10.py

    Args:
        data_dir: Directory containing preprocessed .npy files.
        batch_size: Batch size for training.
        normalize: Whether to normalize to [-1, 1].
        num_workers: Number of data loading workers.
        split: 'unlabeled' (100k, no class filter), 'train' (5k), 'test' (8k),
               or 'train+test' (13k, for labeled with class filtering).
        class_filter: Filter by class (only for labeled splits).
    """

    def __init__(
        self,
        data_dir='./data',
        batch_size=64,
        normalize=True,
        num_workers=4,
        split='unlabeled',
        class_filter: Union[int, str, List[Union[int, str]], None] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.split = split
        self.class_indices = get_class_indices(class_filter)

        if self.class_indices is not None and split == 'unlabeled':
            raise ValueError("Cannot filter by class with 'unlabeled' split (no labels). "
                             "Use split='train+test' for class filtering.")

    def prepare_data(self):
        """Check that preprocessed files exist."""
        splits = ['train', 'test'] if self.split == 'train+test' else [self.split]
        for s in splits:
            path = os.path.join(self.data_dir, f'stl10_gray64_{s}.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Preprocessed file not found: {path}\n"
                    "Run 'python preprocess_stl10.py' first.")

    def _load_split(self, split_name):
        """Load preprocessed .npy files for a split."""
        images = np.load(os.path.join(self.data_dir, f'stl10_gray64_{split_name}.npy'))
        if split_name == 'unlabeled':
            labels = np.full(len(images), -1, dtype=np.int64)
        else:
            labels = np.load(os.path.join(self.data_dir, f'stl10_labels_{split_name}.npy'))
        return images, labels

    def setup(self, stage=None):
        """Setup training dataset from preprocessed cache."""
        if self.split == 'train+test':
            imgs_train, lbls_train = self._load_split('train')
            imgs_test, lbls_test = self._load_split('test')
            all_images = np.concatenate([imgs_train, imgs_test], axis=0)
            all_labels = np.concatenate([lbls_train, lbls_test], axis=0)
        else:
            all_images, all_labels = self._load_split(self.split)

        # Filter by class
        if self.class_indices is not None:
            mask = np.isin(all_labels, self.class_indices)
            all_images = all_images[mask]
            all_labels = all_labels[mask]

        # Normalize to [-1, 1]  (cache is in [0, 1])
        if self.normalize:
            all_images = all_images * 2 - 1

        train_images_t = torch.from_numpy(all_images).float()
        train_labels_t = torch.from_numpy(all_labels).long()

        # Print info
        if self.class_indices is not None:
            class_names = [STL10_CLASSES[i] for i in self.class_indices]
            print(f"Filtered classes: {class_names} (indices: {self.class_indices})")
        print(f"Split: {self.split}, images: {train_images_t.shape}")
        print(f"Range: [{train_images_t.min():.3f}, {train_images_t.max():.3f}]")

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
        """Renormalize data back to [0, 1] for visualization."""
        if self.normalize:
            return (y + 1) / 2
        return y


if __name__ == "__main__":
    print("STL-10 Classes:")
    for idx, name in STL10_CLASSES.items():
        print(f"  {idx}: {name}")

    # Test unlabeled
    print("\n" + "=" * 50)
    print("Testing unlabeled split")
    print("=" * 50)
    dm = STL10GrayDataModule(batch_size=32, split='unlabeled')
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    images, labels = batch
    print(f"Batch shape: {images.shape}")

    # Test labeled with class filter
    print("\n" + "=" * 50)
    print("Testing train+test with class_filter='cat'")
    print("=" * 50)
    dm2 = STL10GrayDataModule(batch_size=32, split='train+test', class_filter='cat')
    dm2.prepare_data()
    dm2.setup()
    batch2 = next(iter(dm2.train_dataloader()))
    images2, labels2 = batch2
    print(f"Batch shape: {images2.shape}")
    print(f"Unique labels: {torch.unique(labels2).tolist()}")
