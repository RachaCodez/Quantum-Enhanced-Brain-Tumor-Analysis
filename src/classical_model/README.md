# Classical Dream Engine - CVAE for Brain Tumor Segmentation

This directory contains the implementation of Stage 1: The Classical "Dream Engine" using a Conditional Variational Autoencoder (CVAE).

## Overview

The CVAE learns the probability distribution **P(segmentation | MRI_image)**, enabling it to generate multiple plausible tumor segmentations for any given MRI scan. This captures the inherent uncertainty in tumor boundaries and creates a "universe of possibilities" that will be interrogated by quantum algorithms in Stage 2.

## Files

### Core Model
- **`cvae.py`**: Complete CVAE architecture
  - `CVAEEncoder`: Encodes MRI + segmentation into latent distribution
  - `CVAEDecoder`: Generates segmentation from latent code + MRI
  - `CVAE`: Full model with reparameterization trick
  - 16.2M parameters (~62 MB)

### Data Processing
- **`dataset.py`**: BraTS dataset loader
  - Loads 4 MRI modalities (FLAIR, T1, T1ce, T2)
  - One-hot encodes 4 segmentation classes
  - Center crops to 128³ volumes
  - Normalizes intensities per modality
  - Creates train/val dataloaders

### Training
- **`train.py`**: Training script
  - `CVAELoss`: Combined Dice + BCE + KL divergence
  - Training loop with validation
  - TensorBoard logging
  - Model checkpointing
  - Learning rate scheduling

### Sampling & Analysis
- **`sampler.py`**: Sampling utilities
  - `CVAESampler`: Generate multiple segmentations
  - Uncertainty quantification (entropy maps)
  - Volume distribution analysis
  - **Multifocality detection** (key clinical question for QAE)

### Utilities
- **`explore_data.py`**: Dataset exploration script

## Dataset

**BraTS (Brain Tumor Segmentation)**
- 484 paired MRI scans and segmentation masks
- Location: `../../data/raw/`
- Format: NIfTI (`.nii.gz`)

**MRI Modalities** (4 channels):
1. FLAIR
2. T1
3. T1ce (T1 contrast-enhanced)
4. T2

**Segmentation Classes** (4 classes):
- Class 0: Background (98.75%)
- Class 1: Necrotic/non-enhancing tumor core (0.59%)
- Class 2: Edema (0.30%)
- Class 3: Enhancing tumor (0.35%)

## Usage

### Quick Start

```python
# Load and explore data
python dataset.py

# Test CVAE architecture
python cvae.py

# Test sampler
python sampler.py
```

### Training

```bash
# Train CVAE model
python train.py
```

Or use the interactive notebook:
```bash
jupyter lab ../../notebooks/train_cvae.ipynb
```

**Training Configuration:**
- Batch size: 2 (adjust based on GPU memory)
- Epochs: 50
- Learning rate: 1e-4
- Latent dimension: 256
- Beta (KL weight): 0.001
- Crop size: 128³

**Hardware Requirements:**
- GPU with 12+ GB VRAM recommended
- Training time: ~6-8 hours on RTX 4070

### Generating Samples

```python
from cvae import CVAE
from sampler import CVAESampler, load_model

# Load trained model
device = torch.device('cuda')
model = load_model('../../models/cvae_*/best_model.pth', device)

# Create sampler
sampler = CVAESampler(model, device)

# Generate samples
samples = sampler.generate_samples(mri_image, num_samples=100)

# Analyze
analysis = sampler.analyze_samples(samples)
print(f"Multifocal probability: {analysis['multifocal_probability']:.3f}")
```

## Model Architecture

### Encoder
```
Input: [MRI (4 ch) + Mask (4 ch)] = 8 channels
    ↓
4x DownBlocks (Conv + Pool)
    ↓
Bottleneck
    ↓
FC layers → (mu, logvar)
```

### Decoder
```
Latent code (z) + MRI condition
    ↓
FC → Reshape
    ↓
4x UpBlocks (Upsample + Skip + Conv)
    ↓
Output: Segmentation (4 classes)
```

### Loss Function

```
Total Loss = Reconstruction + β × KL Divergence

Reconstruction = 0.7 × Dice + 0.3 × BCE
KL Divergence = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
```

## Key Features

1. **Multi-modal Input**: Uses all 4 MRI sequences
2. **One-hot Output**: Probabilistic segmentation across 4 classes
3. **Skip Connections**: U-Net style decoder with condition encoder
4. **Reparameterization**: Enables backpropagation through sampling
5. **Uncertainty Capture**: Latent space encodes segmentation ambiguity

## Clinical Applications

The CVAE "Dream Engine" enables:

1. **Uncertainty Quantification**: Entropy maps show uncertain regions
2. **Volume Estimation**: Distribution of tumor volumes
3. **Multifocality Assessment**: Probability of disconnected tumor components
4. **Treatment Planning**: Multiple plausible tumor extents

## Connection to Stage 2 (Quantum)

The CVAE generates the "universe of possibilities". In Stage 2, we will:

1. Load all samples into quantum superposition
2. Define oracles for clinical queries (e.g., "is_multifocal?")
3. Use Quantum Amplitude Estimation (QAE) to estimate probabilities
4. Achieve **quadratic speedup** over classical counting

**Classical**: O(N) samples needed for ε accuracy
**Quantum**: O(√N) queries needed for ε accuracy

## Next Steps

After training the CVAE:
1. Verify model performance (Dice score > 0.7)
2. Generate sample set for test images
3. Proceed to `../quantum_module/` for QAE implementation
4. Compare classical vs quantum interrogation efficiency

## References

- BraTS Dataset: https://www.med.upenn.edu/cbica/brats/
- CVAE: Sohn et al. "Learning Structured Output Representation using Deep Conditional Generative Models"
- Medical Segmentation: MONAI framework
