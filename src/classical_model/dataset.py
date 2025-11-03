"""
BraTS Dataset and Data Preprocessing Utilities
"""
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS brain tumor segmentation

    Args:
        images_dir: Directory containing MRI images
        labels_dir: Directory containing segmentation labels
        transform: Optional transforms to apply
        crop_size: Size to crop volumes to (depth, height, width)
        normalize: Whether to normalize intensities
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform=None,
        crop_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.crop_size = crop_size
        self.normalize = normalize

        # Get valid file lists (filter out Mac OS metadata)
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if not f.startswith('._') and f.endswith('.nii.gz')
        ])
        self.label_files = sorted([
            f for f in os.listdir(labels_dir)
            if not f.startswith('._') and f.endswith('.nii.gz')
        ])

        assert len(self.image_files) == len(self.label_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.label_files)} labels"

        print(f"Loaded {len(self.image_files)} samples from {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (4, D, H, W) - 4 modalities
            label: Tensor of shape (4, D, H, W) - one-hot encoded 4 classes
        """
        # Load image and label
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        lbl_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = nib.load(img_path).get_fdata()  # (H, W, D, 4)
        label = nib.load(lbl_path).get_fdata()  # (H, W, D)

        # Transpose to (D, H, W, 4) for easier processing
        image = np.transpose(image, (2, 0, 1, 3))  # (D, H, W, 4)
        label = np.transpose(label, (2, 0, 1))      # (D, H, W)

        # Crop to region of interest (center crop)
        image = self._crop_volume(image, self.crop_size)
        label = self._crop_volume(label, self.crop_size)

        # Normalize intensities per modality
        if self.normalize:
            image = self._normalize_image(image)

        # Convert to torch tensors
        # Image: (4, D, H, W)
        image = torch.from_numpy(image).permute(3, 0, 1, 2).float()

        # Label: Convert to one-hot (4, D, H, W)
        label = self._to_one_hot(label, num_classes=4)
        label = torch.from_numpy(label).float()

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def _crop_volume(self, volume: np.ndarray, crop_size: Tuple[int, int, int]) -> np.ndarray:
        """Center crop a volume to the specified size"""
        d, h, w = volume.shape[:3]
        cd, ch, cw = crop_size

        # Calculate crop indices
        d_start = max(0, (d - cd) // 2)
        h_start = max(0, (h - ch) // 2)
        w_start = max(0, (w - cw) // 2)

        d_end = d_start + cd
        h_end = h_start + ch
        w_end = w_start + cw

        if volume.ndim == 4:  # Image with modalities
            return volume[d_start:d_end, h_start:h_end, w_start:w_end, :]
        else:  # Label
            return volume[d_start:d_end, h_start:h_end, w_start:w_end]

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize each modality using z-score normalization"""
        normalized = np.zeros_like(image)
        for i in range(image.shape[-1]):
            modality = image[..., i]
            # Only normalize non-zero regions
            mask = modality > 0
            if mask.sum() > 0:
                mean = modality[mask].mean()
                std = modality[mask].std()
                if std > 0:
                    normalized[..., i] = np.where(
                        mask,
                        (modality - mean) / std,
                        0
                    )
                else:
                    normalized[..., i] = modality
            else:
                normalized[..., i] = modality
        return normalized

    def _to_one_hot(self, label: np.ndarray, num_classes: int = 4) -> np.ndarray:
        """Convert label to one-hot encoding"""
        label_int = label.astype(np.int64)
        one_hot = np.zeros((num_classes,) + label.shape, dtype=np.float32)
        for i in range(num_classes):
            one_hot[i] = (label_int == i).astype(np.float32)
        return one_hot


def get_dataloaders(
    images_dir: str,
    labels_dir: str,
    batch_size: int = 2,
    train_split: float = 0.8,
    num_workers: int = 4,
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders

    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = BraTSDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        crop_size=crop_size,
        normalize=True
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = BraTSDataset(
        images_dir='../../data/raw/imagesTr',
        labels_dir='../../data/raw/labelsTr',
        crop_size=(128, 128, 128)
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load one sample
    image, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Label classes present: {[i for i in range(4) if label[i].sum() > 0]}")
    print(f"  Memory usage: ~{(image.nbytes + label.nbytes) / 1024**2:.2f} MB per sample")
