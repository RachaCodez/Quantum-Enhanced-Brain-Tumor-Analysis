"""
Conditional Variational Autoencoder (CVAE) for Brain Tumor Segmentation

The CVAE learns the distribution P(segmentation | MRI_image), allowing us to:
1. Generate multiple plausible segmentations for a given MRI
2. Capture the inherent uncertainty in tumor boundaries
3. Create a "dream engine" of possible tumor configurations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Standard 3D convolution block with BatchNorm and ReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: Conv -> Conv -> MaxPool"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.pool(x)
        return x, pooled  # Return skip connection and pooled


class UpBlock(nn.Module):
    """Upsampling block: Upsample -> Concat -> Conv -> Conv"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CVAEEncoder(nn.Module):
    """
    Encoder for CVAE - encodes both condition (MRI image) and target (segmentation mask)
    into a latent distribution

    Input: Concatenated [MRI_image (4 channels), segmentation_mask (4 channels)] = 8 channels
    Output: mu and logvar for latent distribution
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Input: 8 channels (4 MRI modalities + 4 segmentation classes)
        self.down1 = DownBlock(8, base_channels)          # 128 -> 64
        self.down2 = DownBlock(base_channels, base_channels * 2)      # 64 -> 32
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)  # 32 -> 16
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)  # 16 -> 8

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # FC layers to latent space
        self.fc_input_size = base_channels * 16 * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Condition (MRI image) - shape (B, 4, D, H, W)
            y: Target (segmentation mask) - shape (B, 4, D, H, W)

        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
        """
        # Concatenate condition and target
        xy = torch.cat([x, y], dim=1)  # (B, 8, D, H, W)

        # Encode
        _, d1 = self.down1(xy)
        _, d2 = self.down2(d1)
        _, d3 = self.down3(d2)
        _, d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)
        pooled = self.adaptive_pool(bottleneck)

        # Flatten
        flat = pooled.view(pooled.size(0), -1)

        # Get latent parameters
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)

        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    Decoder for CVAE - generates segmentation from latent code and condition (MRI image)

    Input: Latent code + condition (MRI image)
    Output: Segmentation mask (4 channels, one per class)
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # FC layer from latent code
        self.fc = nn.Linear(latent_dim, base_channels * 16 * 4 * 4 * 4)

        # Initial upsampling from 4x4x4 to 8x8x8
        self.upsample_init = nn.ConvTranspose3d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )

        # Condition encoder (encodes MRI image for skip connections)
        self.cond_down1 = DownBlock(4, base_channels)          # 128 -> 64
        self.cond_down2 = DownBlock(base_channels, base_channels * 2)      # 64 -> 32
        self.cond_down3 = DownBlock(base_channels * 2, base_channels * 4)  # 32 -> 16
        self.cond_down4 = DownBlock(base_channels * 4, base_channels * 8)  # 16 -> 8

        # Decoder with skip connections from condition
        # (input_channels, skip_channels, output_channels)
        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4)  # 8 -> 16
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)  # 16 -> 32
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels)      # 32 -> 64
        self.up4 = UpBlock(base_channels, base_channels, base_channels)              # 64 -> 128

        # Final output layer
        self.out_conv = nn.Conv3d(base_channels, 4, kernel_size=1)  # 4 classes

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent code (B, latent_dim)
            x: Condition (MRI image) - shape (B, 4, D, H, W)

        Returns:
            Segmentation logits (B, 4, D, H, W)
        """
        # Encode condition to get skip connections
        skip1, c1 = self.cond_down1(x)
        skip2, c2 = self.cond_down2(c1)
        skip3, c3 = self.cond_down3(c2)
        skip4, c4 = self.cond_down4(c3)

        # Decode from latent
        z = self.fc(z)
        z = z.view(z.size(0), self.base_channels * 16, 4, 4, 4)
        z = self.upsample_init(z)  # 8x8x8

        # Upsample with skip connections
        up = self.up1(z, skip4)
        up = self.up2(up, skip3)
        up = self.up3(up, skip2)
        up = self.up4(up, skip1)

        # Final segmentation
        out = self.out_conv(up)

        return out


class CVAE(nn.Module):
    """
    Complete Conditional Variational Autoencoder for Brain Tumor Segmentation

    The "Dream Engine" that learns P(segmentation | MRI_image)
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = CVAEEncoder(latent_dim, base_channels)
        self.decoder = CVAEDecoder(latent_dim, base_channels)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean (B, latent_dim)
            logvar: Log variance (B, latent_dim)

        Returns:
            z: Sampled latent code (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: MRI image (B, 4, D, H, W)
            y: Segmentation mask (B, 4, D, H, W) - only needed during training

        Returns:
            recon: Reconstructed segmentation (B, 4, D, H, W)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """
        if y is not None:
            # Training mode: encode real segmentation
            mu, logvar = self.encoder(x, y)
        else:
            # Inference mode: sample from prior
            batch_size = x.size(0)
            mu = torch.zeros(batch_size, self.latent_dim, device=x.device)
            logvar = torch.zeros(batch_size, self.latent_dim, device=x.device)

        # Sample latent code
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decoder(z, x)

        return recon, mu, logvar

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate multiple segmentation samples for a given MRI image

        Args:
            x: MRI image (B, 4, D, H, W)
            num_samples: Number of samples to generate

        Returns:
            samples: Generated segmentations (B, num_samples, 4, D, H, W)
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            samples = []

            for _ in range(num_samples):
                # Sample from prior N(0, 1)
                z = torch.randn(batch_size, self.latent_dim, device=x.device)

                # Decode
                recon = self.decoder(z, x)
                samples.append(recon)

            samples = torch.stack(samples, dim=1)  # (B, num_samples, 4, D, H, W)

        return samples


def test_cvae():
    """Test CVAE forward and backward pass"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing CVAE on {device}")

    model = CVAE(latent_dim=256, base_channels=16).to(device)

    # Create dummy data
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128, 128).to(device)
    y = torch.randn(batch_size, 4, 128, 128, 128).to(device)

    # Forward pass
    print("\nForward pass...")
    recon, mu, logvar = model(x, y)

    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Logvar shape: {logvar.shape}")

    # Test sampling
    print("\nSampling...")
    samples = model.sample(x, num_samples=5)
    print(f"  Samples shape: {samples.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024**2:.2f} MB (float32)")

    print("\n[SUCCESS] CVAE test passed!")


if __name__ == "__main__":
    test_cvae()
