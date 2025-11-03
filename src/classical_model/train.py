"""
Training script for CVAE model on BraTS dataset
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict
import json
from datetime import datetime

from cvae import CVAE
from dataset import get_dataloaders


class CVAELoss(nn.Module):
    """
    Combined loss for CVAE:
    1. Reconstruction Loss: Dice Loss + BCE for segmentation
    2. KL Divergence: Regularizes latent space to N(0,1)

    Total Loss = Reconstruction + beta * KL_Divergence
    """

    def __init__(self, beta: float = 0.001, dice_weight: float = 0.7):
        super().__init__()
        self.beta = beta
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Dice loss for segmentation

        Args:
            pred: Predicted segmentation logits (B, 4, D, H, W)
            target: Target segmentation one-hot (B, 4, D, H, W)
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice loss (scalar)
        """
        pred = torch.softmax(pred, dim=1)

        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # Compute Dice coefficient per class
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

        # Average over classes and batch
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss

    def bce_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss"""
        return F.binary_cross_entropy_with_logits(pred, target)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between N(mu, var) and N(0, 1)

        KL(N(mu, var) || N(0,1)) = -0.5 * sum(1 + log(var) - mu^2 - var)

        Args:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)

        Returns:
            KL divergence (scalar)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize by batch size
        kl = kl / mu.size(0)
        return kl

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss

        Returns:
            total_loss: Combined loss
            losses: Dictionary of individual loss components
        """
        # Reconstruction losses
        dice = self.dice_loss(pred, target)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        recon_loss = self.dice_weight * dice + self.bce_weight * bce

        # KL divergence
        kl = self.kl_divergence(mu, logvar)

        # Total loss
        total_loss = recon_loss + self.beta * kl

        # Return individual components for logging
        losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'dice': dice.item(),
            'bce': bce.item(),
            'kl': kl.item()
        }

        return total_loss, losses


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Dice score for evaluation

    Args:
        pred: Predicted logits (B, 4, D, H, W)
        target: Target one-hot (B, 4, D, H, W)

    Returns:
        Mean Dice score across all classes
    """
    with torch.no_grad():
        pred = torch.softmax(pred, dim=1)
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice_per_class = (2.0 * intersection + 1.0) / (union + 1.0)

        return dice_per_class.mean().item()


def train_epoch(
    model: CVAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CVAELoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_losses = {'total': 0, 'recon': 0, 'dice': 0, 'bce': 0, 'kl': 0}
    total_dice_score = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        pred, mu, logvar = model(images, labels)

        # Compute loss
        loss, losses = criterion(pred, labels, mu, logvar)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        for key in total_losses:
            total_losses[key] += losses[key]
        total_dice_score += dice_score(pred, labels)

        # Update progress bar
        pbar.set_postfix({
            'loss': losses['total'],
            'dice': total_dice_score / (batch_idx + 1)
        })

    # Average metrics
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    avg_losses['dice_score'] = total_dice_score / num_batches

    return avg_losses


def validate_epoch(
    model: CVAE,
    val_loader: torch.utils.data.DataLoader,
    criterion: CVAELoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()

    total_losses = {'total': 0, 'recon': 0, 'dice': 0, 'bce': 0, 'kl': 0}
    total_dice_score = 0
    num_batches = len(val_loader)

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            pred, mu, logvar = model(images, labels)

            # Compute loss
            loss, losses = criterion(pred, labels, mu, logvar)

            # Track metrics
            for key in total_losses:
                total_losses[key] += losses[key]
            total_dice_score += dice_score(pred, labels)

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'],
                'dice': total_dice_score / (len(pbar.postfix) if pbar.postfix else 1)
            })

    # Average metrics
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    avg_losses['dice_score'] = total_dice_score / num_batches

    return avg_losses


def train(
    model: CVAE,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: str = '../../models',
    beta: float = 0.001,
    scheduler_patience: int = 5
):
    """
    Main training loop

    Args:
        model: CVAE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs to train
        learning_rate: Initial learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        beta: Weight for KL divergence in loss
        scheduler_patience: Patience for learning rate scheduler
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create unique run name
    run_name = f"cvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(os.path.join(run_dir, 'logs'))

    # Loss and optimizer
    criterion = CVAELoss(beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience, factor=0.5
    )

    # Training loop
    best_val_loss = float('inf')
    train_history = []
    val_history = []

    print(f"\nStarting training: {run_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Beta (KL weight): {beta}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_history.append(train_metrics)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        val_history.append(val_metrics)

        # Learning rate scheduling
        scheduler.step(val_metrics['total'])

        # Log to tensorboard
        for key in train_metrics:
            writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
        for key in val_metrics:
            writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['total']:.4f}, Dice: {train_metrics['dice_score']:.4f}")
        print(f"  Val   - Loss: {val_metrics['total']:.4f}, Dice: {val_metrics['dice_score']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(run_dir, 'best_model.pth'))
            print(f"  [SAVED] New best model (val_loss: {best_val_loss:.4f})")

        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['total']
        }
        torch.save(checkpoint, os.path.join(run_dir, 'latest_model.pth'))

    # Save training history
    history = {
        'train': train_history,
        'val': val_history
    }
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {run_dir}")

    return history


if __name__ == "__main__":
    import torch.nn.functional as F

    # Configuration
    CONFIG = {
        'images_dir': '../../data/raw/imagesTr',
        'labels_dir': '../../data/raw/labelsTr',
        'batch_size': 2,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'latent_dim': 256,
        'base_channels': 16,
        'beta': 0.001,  # KL divergence weight
        'crop_size': (128, 128, 128),
        'train_split': 0.8,
        'num_workers': 4,
        'scheduler_patience': 5
    }

    print("Configuration:")
    for key, val in CONFIG.items():
        print(f"  {key}: {val}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        images_dir=CONFIG['images_dir'],
        labels_dir=CONFIG['labels_dir'],
        batch_size=CONFIG['batch_size'],
        train_split=CONFIG['train_split'],
        num_workers=CONFIG['num_workers'],
        crop_size=CONFIG['crop_size']
    )

    # Create model
    print("\nCreating CVAE model...")
    model = CVAE(
        latent_dim=CONFIG['latent_dim'],
        base_channels=CONFIG['base_channels']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        device=device,
        beta=CONFIG['beta'],
        scheduler_patience=CONFIG['scheduler_patience']
    )
