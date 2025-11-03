"""
Sampling utilities for CVAE - the "Dream Engine"

This module provides functions to:
1. Generate multiple plausible segmentations from a trained CVAE
2. Analyze uncertainty across samples
3. Compute clinically relevant statistics (e.g., multifocality probability)
"""
import torch
import numpy as np
from typing import Tuple, List, Dict
import nibabel as nib
from scipy import ndimage

from cvae import CVAE


class CVAESampler:
    """
    Sampler for generating and analyzing multiple segmentation predictions

    This is the "Dream Engine" that generates the universe of plausible
    tumor segmentations for a given MRI image.
    """

    def __init__(self, model: CVAE, device: torch.device):
        """
        Args:
            model: Trained CVAE model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate_samples(
        self,
        mri_image: torch.Tensor,
        num_samples: int = 100
    ) -> np.ndarray:
        """
        Generate multiple segmentation samples for a given MRI image

        Args:
            mri_image: MRI image tensor (1, 4, D, H, W) or (4, D, H, W)
            num_samples: Number of samples to generate

        Returns:
            samples: Array of segmentations (num_samples, 4, D, H, W)
        """
        # Ensure batch dimension
        if mri_image.dim() == 4:
            mri_image = mri_image.unsqueeze(0)

        mri_image = mri_image.to(self.device)

        # Generate samples
        samples = self.model.sample(mri_image, num_samples=num_samples)

        # Convert to class predictions (argmax over class dimension)
        # samples shape: (1, num_samples, 4, D, H, W)
        samples = samples.squeeze(0)  # (num_samples, 4, D, H, W)

        return samples.cpu().numpy()

    def get_class_predictions(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert logits to class predictions

        Args:
            samples: Logits (num_samples, 4, D, H, W)

        Returns:
            predictions: Class labels (num_samples, D, H, W)
        """
        # Softmax and argmax
        probs = np.exp(samples) / np.exp(samples).sum(axis=1, keepdims=True)
        predictions = np.argmax(probs, axis=1)
        return predictions

    def compute_uncertainty_map(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute pixel-wise uncertainty as entropy

        Args:
            samples: Logits (num_samples, 4, D, H, W)

        Returns:
            entropy_map: Uncertainty map (D, H, W)
        """
        # Convert to probabilities
        probs = np.exp(samples) / np.exp(samples).sum(axis=1, keepdims=True)

        # Average probabilities across samples
        mean_probs = probs.mean(axis=0)  # (4, D, H, W)

        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=0)

        return entropy

    def compute_tumor_volume_distribution(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute tumor volume distribution across samples

        Args:
            samples: Logits (num_samples, 4, D, H, W)

        Returns:
            volumes: Dictionary with volumes for each tumor class
        """
        predictions = self.get_class_predictions(samples)

        volumes = {
            'necrotic': [],  # Class 1
            'edema': [],     # Class 2
            'enhancing': [], # Class 3
            'total': []      # Sum of all tumor classes
        }

        for pred in predictions:
            # Count voxels for each class
            necrotic = (pred == 1).sum()
            edema = (pred == 2).sum()
            enhancing = (pred == 3).sum()
            total = necrotic + edema + enhancing

            volumes['necrotic'].append(necrotic)
            volumes['edema'].append(edema)
            volumes['enhancing'].append(enhancing)
            volumes['total'].append(total)

        # Convert to numpy arrays
        for key in volumes:
            volumes[key] = np.array(volumes[key])

        return volumes

    def is_multifocal(self, segmentation: np.ndarray, connectivity: int = 1) -> bool:
        """
        Check if a tumor segmentation is multifocal (multiple disconnected components)

        Args:
            segmentation: Class predictions (D, H, W)
            connectivity: Connectivity for connected components (1 or 2 or 3)

        Returns:
            is_multifocal: True if tumor has multiple disconnected components
        """
        # Create binary mask of tumor (any non-background class)
        tumor_mask = segmentation > 0

        # Find connected components
        labeled, num_components = ndimage.label(tumor_mask, structure=ndimage.generate_binary_structure(3, connectivity))

        # Multifocal if more than one component
        return num_components > 1

    def compute_multifocal_probability(
        self,
        samples: np.ndarray,
        connectivity: int = 1
    ) -> float:
        """
        Compute probability that tumor is multifocal across all samples

        This is a key clinical question that we can answer efficiently
        using the quantum interrogation in Stage 2!

        Args:
            samples: Logits (num_samples, 4, D, H, W)
            connectivity: Connectivity for connected components

        Returns:
            probability: Fraction of samples that are multifocal
        """
        predictions = self.get_class_predictions(samples)

        multifocal_count = 0
        for pred in predictions:
            if self.is_multifocal(pred, connectivity):
                multifocal_count += 1

        probability = multifocal_count / len(predictions)
        return probability

    def analyze_samples(self, samples: np.ndarray) -> Dict:
        """
        Comprehensive analysis of generated samples

        Args:
            samples: Logits (num_samples, 4, D, H, W)

        Returns:
            analysis: Dictionary with various statistics
        """
        predictions = self.get_class_predictions(samples)

        analysis = {
            'num_samples': len(samples),
            'uncertainty_map': self.compute_uncertainty_map(samples),
            'volumes': self.compute_tumor_volume_distribution(samples),
            'multifocal_probability': self.compute_multifocal_probability(samples),
            'predictions': predictions
        }

        # Add volume statistics
        for key in analysis['volumes']:
            vol = analysis['volumes'][key]
            analysis[f'{key}_volume_mean'] = vol.mean()
            analysis[f'{key}_volume_std'] = vol.std()
            analysis[f'{key}_volume_range'] = (vol.min(), vol.max())

        return analysis


def load_model(checkpoint_path: str, device: torch.device, latent_dim: int = 256, base_channels: int = 16) -> CVAE:
    """
    Load a trained CVAE model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        latent_dim: Latent dimension (must match training)
        base_channels: Base channels (must match training)

    Returns:
        model: Loaded CVAE model
    """
    model = CVAE(latent_dim=latent_dim, base_channels=base_channels).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    return model


if __name__ == "__main__":
    """Demo of sampling utilities"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("CVAE Sampler Demo")
    print("=" * 60)

    # Create a dummy model for demonstration
    print("\n1. Creating dummy CVAE model...")
    model = CVAE(latent_dim=256, base_channels=16).to(device)
    model.eval()

    # Create sampler
    print("\n2. Creating sampler...")
    sampler = CVAESampler(model, device)

    # Generate dummy MRI image
    print("\n3. Creating dummy MRI image...")
    dummy_mri = torch.randn(1, 4, 128, 128, 128).to(device)

    # Generate samples
    print("\n4. Generating samples (this may take a moment)...")
    num_samples = 10
    samples = sampler.generate_samples(dummy_mri, num_samples=num_samples)
    print(f"  Generated {num_samples} samples with shape {samples.shape}")

    # Analyze samples
    print("\n5. Analyzing samples...")
    analysis = sampler.analyze_samples(samples)

    print(f"\n  Analysis results:")
    print(f"    Multifocal probability: {analysis['multifocal_probability']:.3f}")
    print(f"    Total tumor volume (mean +/- std): {analysis['total_volume_mean']:.0f} +/- {analysis['total_volume_std']:.0f} voxels")
    print(f"    Total tumor volume range: {analysis['total_volume_range']}")
    print(f"    Uncertainty map shape: {analysis['uncertainty_map'].shape}")

    print("\n[SUCCESS] Sampler demo complete!")
    print("\nThis 'Dream Engine' generates multiple plausible tumor segmentations.")
    print("In Stage 2, we'll use quantum algorithms to efficiently query this")
    print("universe of possibilities for clinical questions like multifocality!")
