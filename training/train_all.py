"""
Training pipeline for Arabic Handwriting Generation models.
Supports CVAE, cGAN, and Transformer-based models.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import wandb
from typing import Dict, List, Optional

# Import models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.preprocessing import ArabicHandwritingDataset, DataLoader
from utils.evaluation import EvaluationMetrics
from models.cvae.model import CVAE, CVAETrainer
from models.cgan.model import CGAN, CGANTrainer
from models.transformer.model import TransformerGAN, TransformerTrainer


class TrainingPipeline:
    """Unified training pipeline for all models."""
    
    def __init__(self, config: dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        self.model_type = config['model_type']
        
        # Create output directory
        self.output_dir = os.path.join('outputs', f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model and trainer
        self.model, self.trainer = self._create_model_and_trainer()
        
        # Initialize evaluation metrics
        self.evaluator = EvaluationMetrics(device)
        
        # Initialize logging
        if config.get('use_wandb', False):
            wandb.init(
                project="arabic-handwriting-generation",
                name=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
    def _create_model_and_trainer(self):
        """Create model and trainer based on model type."""
        if self.model_type == 'cvae':
            model = CVAE(
                vocab_size=self.config.get('vocab_size', 1000),
                image_channels=self.config.get('image_channels', 1),
                latent_dim=self.config.get('latent_dim', 128),
                text_embed_dim=self.config.get('text_embed_dim', 256),
                text_hidden_dim=self.config.get('text_hidden_dim', 512),
                image_size=self.config.get('image_size', 128)
            )
            trainer = CVAETrainer(model, self.device)
            
        elif self.model_type == 'cgan':
            model = CGAN(
                vocab_size=self.config.get('vocab_size', 1000),
                noise_dim=self.config.get('noise_dim', 100),
                image_channels=self.config.get('image_channels', 1),
                text_embed_dim=self.config.get('text_embed_dim', 256),
                text_hidden_dim=self.config.get('text_hidden_dim', 512),
                image_size=self.config.get('image_size', 128)
            )
            trainer = CGANTrainer(model, self.device)
            
        elif self.model_type == 'transformer':
            model = TransformerGAN(
                vocab_size=self.config.get('vocab_size', 1000),
                img_size=self.config.get('image_size', 128),
                patch_size=self.config.get('patch_size', 16),
                embed_dim=self.config.get('embed_dim', 768),
                text_dim=self.config.get('text_dim', 512),
                num_layers=self.config.get('num_layers', 12),
                num_heads=self.config.get('num_heads', 12)
            )
            trainer = TransformerTrainer(model, self.device)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model, trainer
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Train the model."""
        print(f"Starting training for {self.model_type} model...")
        print(f"Training on {len(train_loader)} batches, validation on {len(val_loader)} batches")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 20)
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, epoch)
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model('best_model.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self._save_model(f'checkpoint_epoch_{epoch}.pth')
        
        print("Training completed!")
        return self.history
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.trainer.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss_dict = self.trainer.train_step(batch)
            
            # Accumulate loss
            if self.model_type == 'cvae':
                total_loss += loss_dict['total_loss']
            else:  # GAN models
                total_loss += loss_dict['g_loss'] + loss_dict['d_loss']
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict.get('total_loss', loss_dict.get('g_loss', 0)):.4f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    f'train/{k}': v for k, v in loss_dict.items()
                })
        
        avg_loss = total_loss / num_batches
        return {'avg_loss': avg_loss}
    
    def _validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.trainer.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')
            
            for batch_idx, batch in enumerate(pbar):
                # Validation step
                loss_dict = self.trainer.validate_step(batch)
                
                # Accumulate loss
                if self.model_type == 'cvae':
                    total_loss += loss_dict['total_loss']
                else:  # GAN models
                    total_loss += loss_dict['g_loss'] + loss_dict['d_loss']
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss_dict.get('total_loss', loss_dict.get('g_loss', 0)):.4f}"
                })
        
        avg_loss = total_loss / num_batches
        return {'avg_loss': avg_loss}
    
    def _log_metrics(self, epoch: int, train_loss: Dict, val_loss: Dict):
        """Log training metrics."""
        # Store in history
        self.history['train_loss'].append(train_loss['avg_loss'])
        self.history['val_loss'].append(val_loss['avg_loss'])
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss: {train_loss['avg_loss']:.4f}, "
              f"Val Loss: {val_loss['avg_loss']:.4f}")
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss['avg_loss'],
                'val_loss': val_loss['avg_loss']
            })
    
    def _save_model(self, filename: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.g_optimizer.state_dict() if hasattr(self.trainer, 'g_optimizer') else self.trainer.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate model on test set."""
        print("Evaluating model...")
        
        self.trainer.model.eval()
        
        all_real_images = []
        all_generated_images = []
        all_texts = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch['images'].to(self.device)
                texts = batch['processed_texts']
                
                # Generate images
                if self.model_type == 'cvae':
                    generated = self.trainer.generate_samples(texts, num_samples=1)
                else:
                    generated = self.trainer.generate_samples(texts, num_samples=1)
                
                all_real_images.append(images.cpu())
                all_generated_images.append(generated.cpu())
                all_texts.extend(texts)
        
        # Concatenate all images
        real_images = torch.cat(all_real_images, dim=0)
        generated_images = torch.cat(all_generated_images, dim=0)
        
        # Calculate metrics
        metrics = self.evaluator.evaluate_all_metrics(real_images, generated_images, all_texts)
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Evaluation Metrics:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")
        
        return metrics
    
    def generate_samples(self, texts: List[str], num_samples: int = 4) -> torch.Tensor:
        """Generate samples for given texts."""
        self.trainer.model.eval()
        
        with torch.no_grad():
            generated = self.trainer.generate_samples(texts, num_samples)
        
        return generated
    
    def plot_training_history(self):
        """Plot training history."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.model_type.upper()} Training History')
        ax.legend()
        ax.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def create_config(model_type: str, **kwargs) -> dict:
    """Create configuration for training."""
    base_config = {
        'model_type': model_type,
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 20,
        'save_interval': 10,
        'use_wandb': False,
        'vocab_size': 1000,
        'image_channels': 1,
        'image_size': 128,
        'text_embed_dim': 256,
        'text_hidden_dim': 512,
    }
    
    # Model-specific configs
    if model_type == 'cvae':
        base_config.update({
            'latent_dim': 128,
        })
    elif model_type == 'cgan':
        base_config.update({
            'noise_dim': 100,
        })
    elif model_type == 'transformer':
        base_config.update({
            'patch_size': 16,
            'embed_dim': 768,
            'text_dim': 512,
            'num_layers': 12,
            'num_heads': 12,
        })
    
    # Update with user-provided config
    base_config.update(kwargs)
    
    return base_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Arabic Handwriting Generation Models')
    parser.add_argument('--model', type=str, choices=['cvae', 'cgan', 'transformer'], 
                       required=True, help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='data/raw', 
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create configuration
    config = create_config(
        model_type=args.model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = ArabicHandwritingDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("No data found. Creating placeholder dataset...")
        from utils.preprocessing import create_placeholder_dataset
        create_placeholder_dataset(args.data_dir, 100)
        dataset = ArabicHandwritingDataset(args.data_dir)
    
    # Create data loaders
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    train_indices, val_indices, test_indices = data_loader.split_dataset()
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create training pipeline
    pipeline = TrainingPipeline(config, device)
    
    # Train model
    history = pipeline.train(train_loader, val_loader, args.num_epochs)
    
    # Evaluate model
    metrics = pipeline.evaluate(test_loader)
    
    # Plot training history
    pipeline.plot_training_history()
    
    # Generate sample images
    sample_texts = ["مرحبا بالعالم", "أهلا وسهلا", "شكرا لك", "مع السلامة"]
    generated_samples = pipeline.generate_samples(sample_texts, num_samples=2)
    
    # Save sample images
    from utils.evaluation import visualize_comparison
    visualize_comparison(
        torch.randn(len(sample_texts), 1, 128, 128),  # Dummy real images
        generated_samples[:len(sample_texts)],
        num_samples=len(sample_texts),
        save_path=os.path.join(pipeline.output_dir, 'sample_generations.png')
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()

