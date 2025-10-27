"""
Test script to verify the Arabic Handwriting Generation project setup.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test if all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from utils.preprocessing import ArabicHandwritingDataset, DataLoader, create_placeholder_dataset
        print("✓ Preprocessing utilities imported successfully")
    except Exception as e:
        print(f"✗ Error importing preprocessing utilities: {e}")
        return False
    
    try:
        from utils.evaluation import EvaluationMetrics
        print("✓ Evaluation metrics imported successfully")
    except Exception as e:
        print(f"✗ Error importing evaluation metrics: {e}")
        return False
    
    try:
        from utils.comparison import ModelComparator
        print("✓ Comparison tools imported successfully")
    except Exception as e:
        print(f"✗ Error importing comparison tools: {e}")
        return False
    
    try:
        from models.cvae.model import CVAE, CVAETrainer
        print("✓ CVAE model imported successfully")
    except Exception as e:
        print(f"✗ Error importing CVAE model: {e}")
        return False
    
    try:
        from models.cgan.model import CGAN, CGANTrainer
        print("✓ cGAN model imported successfully")
    except Exception as e:
        print(f"✗ Error importing cGAN model: {e}")
        return False
    
    try:
        from models.transformer.model import TransformerGAN, TransformerTrainer
        print("✓ Transformer model imported successfully")
    except Exception as e:
        print(f"✗ Error importing Transformer model: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created successfully."""
    print("\nTesting model creation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Test CVAE
        cvae = CVAE(vocab_size=1000, latent_dim=128, image_size=128)
        cvae_trainer = CVAETrainer(cvae, device)
        print("✓ CVAE model and trainer created successfully")
        
        # Test cGAN
        cgan = CGAN(vocab_size=1000, noise_dim=100, image_size=128)
        cgan_trainer = CGANTrainer(cgan, device)
        print("✓ cGAN model and trainer created successfully")
        
        # Test Transformer
        transformer = TransformerGAN(vocab_size=1000, img_size=128, patch_size=16)
        transformer_trainer = TransformerTrainer(transformer, device)
        print("✓ Transformer model and trainer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating models: {e}")
        return False

def test_data_processing():
    """Test data processing pipeline."""
    print("\nTesting data processing...")
    
    try:
        # Create placeholder dataset
        from utils.preprocessing import create_placeholder_dataset
        create_placeholder_dataset('data/raw', 10)
        print("✓ Placeholder dataset created successfully")
        
        # Load dataset
        from utils.preprocessing import ArabicHandwritingDataset
        dataset = ArabicHandwritingDataset('data/raw')
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Test data loader
        from utils.preprocessing import DataLoader
        data_loader = DataLoader(dataset, batch_size=4)
        loader = data_loader.create_dataloader()
        
        # Test a batch
        for batch in loader:
            print(f"✓ Batch loaded successfully: images shape {batch['images'].shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data processing: {e}")
        return False

def test_evaluation():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    try:
        from utils.evaluation import EvaluationMetrics
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluator = EvaluationMetrics(device)
        
        # Create dummy data
        real_images = torch.randn(4, 1, 128, 128).to(device)
        generated_images = torch.randn(4, 1, 128, 128).to(device)
        
        # Test metrics calculation
        metrics = evaluator.evaluate_all_metrics(real_images, generated_images)
        print(f"✓ Evaluation metrics calculated successfully: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        return False

def test_model_inference():
    """Test model inference."""
    print("\nTesting model inference...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test CVAE inference
        from models.cvae.model import CVAE, CVAETrainer
        cvae = CVAE(vocab_size=1000, latent_dim=128, image_size=128)
        cvae_trainer = CVAETrainer(cvae, device)
        
        # Generate samples
        texts = ["مرحبا", "أهلا"]
        generated = cvae_trainer.generate_samples(texts, num_samples=2)
        print(f"✓ CVAE inference successful: generated shape {generated.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model inference: {e}")
        return False

def main():
    """Run all tests."""
    print("Arabic Handwriting Generation - Project Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_processing,
        test_evaluation,
        test_model_inference
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

