"""
Comparative analysis and visualization tools for Arabic Handwriting Generation.
Provides comprehensive comparison of different models and their performance.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import cv2
from PIL import Image

# Import models and utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cvae.model import CVAE, CVAETrainer
from models.cgan.model import CGAN, CGANTrainer
from models.transformer.model import TransformerGAN, TransformerTrainer
from utils.evaluation import EvaluationMetrics, visualize_comparison, plot_metrics_comparison
from utils.preprocessing import ArabicTextProcessor


class ModelComparator:
    """Comprehensive model comparison tool."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.results = {}
        self.text_processor = ArabicTextProcessor()
        self.evaluator = EvaluationMetrics(device)
        
    def load_models(self, model_paths: Dict[str, str]):
        """Load multiple trained models."""
        for model_type, model_path in model_paths.items():
            if os.path.exists(model_path):
                self._load_single_model(model_type, model_path)
            else:
                print(f"Warning: Model path {model_path} not found for {model_type}")
    
    def _load_single_model(self, model_type: str, model_path: str):
        """Load a single model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        if model_type == 'cvae':
            model = CVAE(
                vocab_size=config.get('vocab_size', 1000),
                image_channels=config.get('image_channels', 1),
                latent_dim=config.get('latent_dim', 128),
                text_embed_dim=config.get('text_embed_dim', 256),
                text_hidden_dim=config.get('text_hidden_dim', 512),
                image_size=config.get('image_size', 128)
            )
            trainer = CVAETrainer(model, self.device)
            
        elif model_type == 'cgan':
            model = CGAN(
                vocab_size=config.get('vocab_size', 1000),
                noise_dim=config.get('noise_dim', 100),
                image_channels=config.get('image_channels', 1),
                text_embed_dim=config.get('text_embed_dim', 256),
                text_hidden_dim=config.get('text_hidden_dim', 512),
                image_size=config.get('image_size', 128)
            )
            trainer = CGANTrainer(model, self.device)
            
        elif model_type == 'transformer':
            model = TransformerGAN(
                vocab_size=config.get('vocab_size', 1000),
                img_size=config.get('image_size', 128),
                patch_size=config.get('patch_size', 16),
                embed_dim=config.get('embed_dim', 768),
                text_dim=config.get('text_dim', 512),
                num_layers=config.get('num_layers', 12),
                num_heads=config.get('num_heads', 12)
            )
            trainer = TransformerTrainer(model, self.device)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.models[model_type] = {
            'model': model,
            'trainer': trainer,
            'config': config,
            'history': checkpoint.get('history', {})
        }
        
        print(f"Loaded {model_type} model from {model_path}")
    
    def generate_comparison_samples(self, texts: List[str], num_samples: int = 4) -> Dict[str, List[np.ndarray]]:
        """Generate samples from all models for comparison."""
        results = {}
        
        for model_type, model_info in self.models.items():
            trainer = model_info['trainer']
            model_results = []
            
            for text in texts:
                try:
                    # Preprocess text
                    processed_text = self.text_processor.preprocess_text(text)
                    
                    # Generate samples
                    generated = trainer.generate_samples([processed_text], num_samples)
                    
                    # Convert to numpy arrays
                    images = []
                    for i in range(generated.shape[0]):
                        img = generated[i].squeeze().cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        images.append(img)
                    
                    model_results.append(images)
                    
                except Exception as e:
                    print(f"Error generating with {model_type} for text '{text}': {e}")
                    model_results.append([])
            
            results[model_type] = model_results
        
        return results
    
    def evaluate_models(self, test_loader) -> Dict[str, Dict[str, float]]:
        """Evaluate all loaded models on test data."""
        results = {}
        
        for model_type, model_info in self.models.items():
            print(f"Evaluating {model_type} model...")
            
            trainer = model_info['trainer']
            trainer.model.eval()
            
            all_real_images = []
            all_generated_images = []
            all_texts = []
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['images'].to(self.device)
                    texts = batch['processed_texts']
                    
                    # Generate images
                    generated = trainer.generate_samples(texts, num_samples=1)
                    
                    all_real_images.append(images.cpu())
                    all_generated_images.append(generated.cpu())
                    all_texts.extend(texts)
            
            # Concatenate all images
            real_images = torch.cat(all_real_images, dim=0)
            generated_images = torch.cat(all_generated_images, dim=0)
            
            # Calculate metrics
            metrics = self.evaluator.evaluate_all_metrics(real_images, generated_images, all_texts)
            results[model_type] = metrics
            
            print(f"{model_type} evaluation completed")
        
        return results
    
    def create_comparison_visualization(self, texts: List[str], num_samples: int = 4, 
                                      save_path: str = None) -> plt.Figure:
        """Create comprehensive comparison visualization."""
        # Generate samples from all models
        results = self.generate_comparison_samples(texts, num_samples)
        
        # Create figure
        n_texts = len(texts)
        n_models = len(self.models)
        
        fig, axes = plt.subplots(n_texts, n_models + 1, figsize=(4 * (n_models + 1), 3 * n_texts))
        
        if n_texts == 1:
            axes = axes.reshape(1, -1)
        
        # Plot for each text
        for text_idx, text in enumerate(texts):
            # First column: Original text
            axes[text_idx, 0].text(0.5, 0.5, text, ha='center', va='center', 
                                 fontsize=12, transform=axes[text_idx, 0].transAxes)
            axes[text_idx, 0].set_title('Input Text', fontsize=10)
            axes[text_idx, 0].axis('off')
            
            # Other columns: Generated samples
            model_idx = 0
            for model_type, model_results in results.items():
                if model_results[text_idx]:
                    # Show first sample
                    img = model_results[text_idx][0]
                    axes[text_idx, model_idx + 1].imshow(img, cmap='gray')
                    axes[text_idx, model_idx + 1].set_title(f'{model_type.upper()}', fontsize=10)
                else:
                    axes[text_idx, model_idx + 1].text(0.5, 0.5, 'Error', ha='center', va='center',
                                                     transform=axes[text_idx, model_idx + 1].transAxes)
                    axes[text_idx, model_idx + 1].set_title(f'{model_type.upper()}', fontsize=10)
                
                axes[text_idx, model_idx + 1].axis('off')
                model_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, metrics_results: Dict[str, Dict[str, float]], 
                              save_path: str = None) -> plt.Figure:
        """Plot comparison of evaluation metrics."""
        return plot_metrics_comparison(metrics_results, save_path)
    
    def plot_training_curves(self, save_path: str = None) -> plt.Figure:
        """Plot training curves for all models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['train_loss', 'val_loss']
        if len(axes) > 2:
            metrics.extend(['g_loss', 'd_loss'])
        
        for i, metric in enumerate(metrics[:len(axes)]):
            ax = axes[i]
            
            for model_type, model_info in self.models.items():
                history = model_info.get('history', {})
                if metric in history and history[metric]:
                    epochs = range(1, len(history[metric]) + 1)
                    ax.plot(epochs, history[metric], label=f'{model_type.upper()}', marker='o', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_table(self, metrics_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create performance comparison table."""
        # Convert to DataFrame
        df = pd.DataFrame(metrics_results).T
        
        # Round to 4 decimal places
        df = df.round(4)
        
        # Add ranking columns
        for metric in df.columns:
            if metric in ['fid', 'l1_loss', 'l2_loss']:
                # Lower is better
                df[f'{metric}_rank'] = df[metric].rank(ascending=True)
            else:
                # Higher is better
                df[f'{metric}_rank'] = df[metric].rank(ascending=False)
        
        return df
    
    def generate_comparison_report(self, texts: List[str], test_loader=None, 
                                 output_dir: str = 'comparison_results') -> str:
        """Generate comprehensive comparison report."""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'comparison_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Arabic Handwriting Generation - Model Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model information
            f.write("Loaded Models:\n")
            for model_type, model_info in self.models.items():
                config = model_info['config']
                f.write(f"- {model_type.upper()}: {config.get('image_size', 128)}x{config.get('image_size', 128)} images\n")
            f.write("\n")
            
            # Evaluation metrics
            if test_loader:
                f.write("Evaluation Metrics:\n")
                metrics_results = self.evaluate_models(test_loader)
                
                # Create performance table
                df = self.create_performance_table(metrics_results)
                f.write(df.to_string())
                f.write("\n\n")
                
                # Best model for each metric
                f.write("Best Model for Each Metric:\n")
                for metric in df.columns:
                    if not metric.endswith('_rank'):
                        best_model = df[metric].idxmin() if metric in ['fid', 'l1_loss', 'l2_loss'] else df[metric].idxmax()
                        f.write(f"- {metric}: {best_model.upper()}\n")
                f.write("\n")
            
            # Sample generation results
            f.write("Sample Generation Results:\n")
            results = self.generate_comparison_samples(texts, 4)
            
            for text in texts:
                f.write(f"\nText: {text}\n")
                for model_type, model_results in results.items():
                    if model_results[texts.index(text)]:
                        f.write(f"- {model_type.upper()}: Generated {len(model_results[texts.index(text)])} samples\n")
                    else:
                        f.write(f"- {model_type.upper()}: Generation failed\n")
        
        # Generate visualizations
        self.create_comparison_visualization(
            texts, 4, 
            os.path.join(output_dir, 'sample_comparison.png')
        )
        
        if test_loader:
            metrics_results = self.evaluate_models(test_loader)
            self.plot_metrics_comparison(
                metrics_results,
                os.path.join(output_dir, 'metrics_comparison.png')
            )
        
        self.plot_training_curves(
            os.path.join(output_dir, 'training_curves.png')
        )
        
        print(f"Comparison report generated at: {report_path}")
        return report_path


def analyze_model_performance(model_paths: Dict[str, str], test_data_path: str, 
                            output_dir: str = 'analysis_results'):
    """Analyze and compare model performance."""
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load models
    comparator.load_models(model_paths)
    
    if not comparator.models:
        print("No models loaded. Please check model paths.")
        return
    
    # Load test data
    from utils.preprocessing import ArabicHandwritingDataset, DataLoader
    
    test_dataset = ArabicHandwritingDataset(test_data_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Sample texts for comparison
    sample_texts = [
        "مرحبا بالعالم",
        "أهلا وسهلا",
        "شكرا لك",
        "مع السلامة",
        "صباح الخير"
    ]
    
    # Generate comprehensive report
    report_path = comparator.generate_comparison_report(sample_texts, test_loader, output_dir)
    
    print("Analysis completed!")
    print(f"Results saved in: {output_dir}")
    
    return comparator


def create_style_consistency_analysis(model_paths: Dict[str, str], 
                                    reference_texts: List[str],
                                    output_dir: str = 'style_analysis'):
    """Analyze style consistency across different texts."""
    
    comparator = ModelComparator()
    comparator.load_models(model_paths)
    
    if not comparator.models:
        print("No models loaded. Please check model paths.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples for each text
    results = comparator.generate_comparison_samples(reference_texts, 8)
    
    # Analyze style consistency
    consistency_analysis = {}
    
    for model_type, model_results in results.items():
        model_consistency = []
        
        for text_idx, text_samples in enumerate(model_results):
            if text_samples:
                # Calculate pairwise similarities within samples for same text
                similarities = []
                for i in range(len(text_samples)):
                    for j in range(i + 1, len(text_samples)):
                        # Calculate SSIM between samples
                        from skimage.metrics import structural_similarity as ssim
                        similarity = ssim(text_samples[i], text_samples[j], data_range=255)
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities) if similarities else 0
                model_consistency.append(avg_similarity)
        
        consistency_analysis[model_type] = {
            'mean_consistency': np.mean(model_consistency),
            'std_consistency': np.std(model_consistency),
            'per_text_consistency': model_consistency
        }
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'style_consistency_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(consistency_analysis, f, indent=2, ensure_ascii=False)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = list(consistency_analysis.keys())
    mean_consistencies = [consistency_analysis[model]['mean_consistency'] for model in models]
    std_consistencies = [consistency_analysis[model]['std_consistency'] for model in models]
    
    bars = ax.bar(models, mean_consistencies, yerr=std_consistencies, capsize=5)
    ax.set_ylabel('Style Consistency Score')
    ax.set_title('Style Consistency Analysis Across Models')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, mean_consistencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'style_consistency.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Style consistency analysis completed!")
    print(f"Results saved in: {output_dir}")
    
    return consistency_analysis


if __name__ == "__main__":
    # Example usage
    model_paths = {
        'cvae': 'outputs/cvae_20231201_120000/best_model.pth',
        'cgan': 'outputs/cgan_20231201_120000/best_model.pth',
        'transformer': 'outputs/transformer_20231201_120000/best_model.pth'
    }
    
    # Analyze performance
    comparator = analyze_model_performance(
        model_paths=model_paths,
        test_data_path='data/raw',
        output_dir='analysis_results'
    )
    
    # Analyze style consistency
    reference_texts = [
        "مرحبا بالعالم",
        "أهلا وسهلا",
        "شكرا لك"
    ]
    
    consistency_analysis = create_style_consistency_analysis(
        model_paths=model_paths,
        reference_texts=reference_texts,
        output_dir='style_analysis'
    )

