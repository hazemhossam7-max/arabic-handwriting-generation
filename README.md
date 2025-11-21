# Arabic Handwriting Generation Using Conditional Deep Generative Models

A comprehensive implementation of three different deep generative approaches for synthesizing realistic Arabic handwritten text. This project compares Conditional Variational Autoencoder (CVAE), Conditional Generative Adversarial Network (cGAN), and Transformer-based generators for Arabic handwriting generation.

## 🎯 Project Overview

This project implements and compares three state-of-the-art deep generative models for Arabic handwriting synthesis:

1. **Conditional Variational Autoencoder (CVAE)** - Provides interpretable latent space and stable training
2. **Conditional Generative Adversarial Network (cGAN)** - Pix2Pix-style architecture for high-quality generation
3. **Transformer-based Generator** - Vision Transformer with autoregressive decoding for cutting-edge results

## 📁 Project Structure

```
arabic_handwriting_generation/
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Preprocessed data
│   └── augmentation/           # Augmented samples
├── models/
│   ├── cvae/
│   │   └── model.py            # CVAE implementation
│   ├── cgan/
│   │   └── model.py            # cGAN implementation
│   ├── transformer/
│   │   └── model.py            # Transformer implementation
│   └── diffusion/              # Diffusion model (bonus)
├── utils/
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── evaluation.py          # Evaluation metrics (FID, SSIM, PSNR)
│   └── comparison.py           # Model comparison tools
├── training/
│   └── train_all.py           # Unified training pipeline
├── inference/
│   └── demo_interface.py       # Gradio/Streamlit demo interface
├── configs/
│   ├── cvae_config.yaml        # CVAE configuration
│   ├── cgan_config.yaml        # cGAN configuration
│   └── transformer_config.yaml # Transformer configuration
├── notebooks/
│   └── model_comparison.ipynb  # Comprehensive analysis notebook
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd arabic-handwriting-generation

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Create placeholder dataset for testing
python -c "from utils.preprocessing import create_placeholder_dataset; create_placeholder_dataset('data/raw', 100)"
```

### 3. Train Models

```bash
# Train CVAE
python training/train_all.py --model cvae --data_dir data/raw --num_epochs 50

# Train cGAN
python training/train_all.py --model cgan --data_dir data/raw --num_epochs 50

# Train Transformer
python training/train_all.py --model transformer --data_dir data/raw --num_epochs 50
```

### 4. Run Demo Interface

```bash
# Gradio interface
python inference/demo_interface.py --interface gradio

# Streamlit interface
streamlit run inference/demo_interface.py -- --interface streamlit
```

## 📊 Model Comparison

| Model | Parameters | Size (MB) | Training Stability | Generation Quality | Inference Speed |
|-------|------------|-----------|-------------------|-------------------|-----------------|
| CVAE | ~2.1M | ~8.4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| cGAN | ~3.8M | ~15.2 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Transformer | ~86.4M | ~345.6 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 🔧 Detailed Usage

### Training Individual Models

```bash
# CVAE with custom parameters
python training/train_all.py \
    --model cvae \
    --data_dir data/raw \
    --batch_size 32 \
    --num_epochs 100 \
    --use_wandb

# cGAN with custom parameters
python training/train_all.py \
    --model cgan \
    --data_dir data/raw \
    --batch_size 16 \
    --num_epochs 100

# Transformer with custom parameters
python training/train_all.py \
    --model transformer \
    --data_dir data/raw \
    --batch_size 8 \
    --num_epochs 100
```

### Model Evaluation

```python
from utils.comparison import ModelComparator

# Load trained models
model_paths = {
    'cvae': 'outputs/cvae_20231201_120000/best_model.pth',
    'cgan': 'outputs/cgan_20231201_120000/best_model.pth',
    'transformer': 'outputs/transformer_20231201_120000/best_model.pth'
}

# Create comparator
comparator = ModelComparator()
comparator.load_models(model_paths)

# Generate comparison report
comparator.generate_comparison_report(
    texts=["مرحبا بالعالم", "أهلا وسهلا", "شكرا لك"],
    test_loader=test_loader,
    output_dir='comparison_results'
)
```

### Custom Text Generation

```python
from models.cvae.model import CVAE, CVAETrainer

# Load trained model
model = CVAE(vocab_size=1000, latent_dim=128)
checkpoint = torch.load('path/to/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate handwriting
trainer = CVAETrainer(model)
generated_images = trainer.generate_samples(["مرحبا بالعالم"], num_samples=4)
```

## 📈 Evaluation Metrics

The project implements comprehensive evaluation metrics:

- **FID (Fréchet Inception Distance)** - Measures quality and diversity
- **SSIM (Structural Similarity Index)** - Measures structural similarity
- **PSNR (Peak Signal-to-Noise Ratio)** - Measures reconstruction quality
- **L1/L2 Loss** - Pixel-wise reconstruction errors
- **Text Alignment Score** - OCR-based text accuracy (placeholder)
- **Style Consistency Score** - Measures handwriting style consistency

## 🎨 Demo Interface Features

The demo interface provides:

1. **Single Model Generation** - Generate samples using one model
2. **Model Comparison** - Compare outputs from all loaded models
3. **Style Transfer Evaluation** - Analyze style consistency
4. **Real-time Generation** - Interactive text input and generation
5. **Batch Processing** - Generate multiple samples at once

## 📚 Dataset Support

The project supports multiple Arabic handwriting datasets:

- **KHATT**: https://www.kaggle.com/datasets/nizarcharrada/khattarabic
- **AHCD**: https://www.kaggle.com/datasets/skanderkammoun/ahcd-arabic-handwriting
- **Additional datasets** as specified in the project description

## 🔬 Research Applications

This implementation can be used for:

- **Document Synthesis** - Generate realistic handwritten documents
- **Digital Forensics** - Analyze handwriting patterns
- **Educational Tools** - Create handwriting practice materials
- **Accessibility** - Convert text to handwritten form
- **Artistic Applications** - Create handwritten art and calligraphy

## 🛠️ Technical Details

### Model Architectures

1. **CVAE**: Encoder-decoder with variational latent space and text conditioning
2. **cGAN**: Pix2Pix-style generator with conditional discriminator
3. **Transformer**: Vision Transformer with patch-based processing and autoregressive generation

### Key Features

- **Arabic Text Processing** - Proper reshaping and bidirectional text handling
- **Data Augmentation** - Rotation, distortion, and noise augmentation
- **Multi-GPU Support** - Automatic device detection and GPU utilization
- **Comprehensive Logging** - Weights & Biases integration for experiment tracking
- **Modular Design** - Easy to extend and modify individual components
