"""
Preprocess STL-10: RGB 96x96 -> Grayscale 64x64, save as .npy cache.

Run once after downloading the dataset:
    python preprocess_stl10.py

Creates:
    data/stl10_gray64_unlabeled.npy   (100000, 1, 64, 64) float32 [0, 1]
    data/stl10_gray64_train.npy       (5000, 1, 64, 64)
    data/stl10_gray64_test.npy        (8000, 1, 64, 64)
    data/stl10_labels_train.npy       (5000,) int64
    data/stl10_labels_test.npy        (8000,) int64
"""

import os
import numpy as np
import torch
import torchvision

DATA_DIR = './data/stl10'
OUT_DIR = './data'
IMAGE_SIZE = 64


def rgb_to_gray_and_resize(rgb_images, size=IMAGE_SIZE):
    """(N, 3, 96, 96) uint8 -> (N, 1, size, size) float32 [0,1]"""
    images_t = torch.from_numpy(rgb_images).float() / 255.0
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    gray = (images_t * weights).sum(dim=1, keepdim=True)
    if size != 96:
        gray = torch.nn.functional.interpolate(
            gray, size=size, mode='bilinear', antialias=True)
    return gray.numpy()


def process_split(split_name):
    print(f"Processing {split_name}...")
    dataset = torchvision.datasets.STL10(root=DATA_DIR, split=split_name, download=False)
    images = dataset.data  # (N, 3, 96, 96) uint8
    labels = np.array(dataset.labels)  # (N,)

    # Process in batches to limit memory
    batch_size = 10000
    n = len(images)
    results = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = rgb_to_gray_and_resize(images[i:end])
        results.append(batch)
        print(f"  {end}/{n}")

    gray = np.concatenate(results, axis=0)  # (N, 1, 64, 64)
    print(f"  Shape: {gray.shape}, range: [{gray.min():.4f}, {gray.max():.4f}]")

    # Save
    np.save(os.path.join(OUT_DIR, f'stl10_gray64_{split_name}.npy'), gray)
    if split_name != 'unlabeled':
        np.save(os.path.join(OUT_DIR, f'stl10_labels_{split_name}.npy'), labels)

    return gray, labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ['unlabeled', 'train', 'test']:
        process_split(split)

    print("\nAll preprocessed files saved to data/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith('stl10_'):
            path = os.path.join(OUT_DIR, f)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
