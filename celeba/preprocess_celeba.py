"""
Preprocess CelebA: RGB 178x218 -> center crop -> Grayscale 64x64 and 128x128.

CelebA must be downloaded manually (Google Drive auth required):
    1. Download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
       or use: pip install gdown && gdown 0B7EVK8r0v71pZjFTYXZWM3FlRnM
    2. Extract to data/celeba/img_align_celeba/

Run once after preparing the data:
    python preprocess_celeba.py

Creates:
    data/celeba_gray64.npy    (202599, 1, 64, 64)   float32 [0, 1]
    data/celeba_gray128.npy   (202599, 1, 128, 128)  float32 [0, 1]
"""

import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATA_DIR = './data/celeba'
IMG_DIR = os.path.join(DATA_DIR, 'img_align_celeba')
OUT_DIR = './data'


def process_images(image_size, crop_size=178):
    """Load, center-crop, resize, grayscale all CelebA images.

    Args:
        image_size: Target size (64 or 128).
        crop_size: Center crop size before resize.
            178 = full width (crop height only, preserves hair/background).
            140 = tight face crop (DDPM/Score SDE convention).

    Returns:
        numpy array (N, 1, image_size, image_size) float32 [0, 1]
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS,
                          antialias=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # -> [0, 1]
    ])

    filenames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    print(f"Found {len(filenames)} images")

    # Process in chunks
    chunk_size = 10000
    results = []

    for start in tqdm(range(0, len(filenames), chunk_size),
                      desc=f"Processing {image_size}x{image_size}"):
        end = min(start + chunk_size, len(filenames))
        batch = []
        for fname in filenames[start:end]:
            img = Image.open(os.path.join(IMG_DIR, fname))
            tensor = transform(img)  # (1, image_size, image_size)
            batch.append(tensor.numpy())
        results.append(np.stack(batch))

    return np.concatenate(results, axis=0)


def main():
    if not os.path.isdir(IMG_DIR):
        print(f"Error: {IMG_DIR} not found.")
        print("Please download CelebA img_align_celeba.zip and extract to data/celeba/")
        print("  wget --no-check-certificate 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&export=download' -O img_align_celeba.zip")
        print("  unzip img_align_celeba.zip -d data/celeba/")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    for size in [64, 128]:
        print(f"\n=== Processing {size}x{size} ===")
        images = process_images(size)
        print(f"Shape: {images.shape}, range: [{images.min():.4f}, {images.max():.4f}]")

        out_path = os.path.join(OUT_DIR, f'celeba_gray{size}.npy')
        np.save(out_path, images)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved: {out_path} ({size_mb:.0f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
