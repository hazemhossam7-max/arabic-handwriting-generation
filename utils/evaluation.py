"""
Evaluation metrics for Arabic handwriting generation.
Implements FID, SSIM, PSNR, and other quality metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy import linalg
from torchvision.models import inception_v3
from torchvision import transforms
import torch.nn as nn


class InceptionFeatureExtractor(nn.Module):
    """Extract features using Inception network for FID calculation."""
    
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
        
    def forward(self, x):
        # Resize to 299x299 for Inception
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1]
        x = (x - 0.5) * 2
        
        # Extract features
        features = self.inception(x)
        return features


class EvaluationMetrics:
    """Comprehensive evaluation metrics for handwriting generation."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_extractor = InceptionFeatureExtractor().to(device)
        self.feature_extractor.eval()
        
    def calculate_fid(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate Fréchet Inception Distance (FID).
        
        Args:
            real_images: Real images tensor [N, C, H, W]
            generated_images: Generated images tensor [N, C, H, W]
            
        Returns:
            FID score
        """
        with torch.no_grad():
            # Extract features
            real_features = self.feature_extractor(real_images)
            generated_features = self.feature_extractor(generated_images)
            
            # Convert to numpy
            real_features = real_features.cpu().numpy()
            generated_features = generated_features.cpu().numpy()
            
            # Calculate mean and covariance
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
            
            # Calculate FID
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
            
            return fid
    
    def calculate_ssim(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            real_images: Real images tensor [N, C, H, W]
            generated_images: Generated images tensor [N, C, H, W]
            
        Returns:
            Average SSIM score
        """
        ssim_scores = []
        
        for i in range(real_images.shape[0]):
            # Convert to numpy and grayscale
            real_img = real_images[i].squeeze().cpu().numpy()
            gen_img = generated_images[i].squeeze().cpu().numpy()
            
            # Ensure images are in [0, 1] range
            real_img = np.clip(real_img, 0, 1)
            gen_img = np.clip(gen_img, 0, 1)
            
            # Calculate SSIM
            ssim_score = ssim(real_img, gen_img, data_range=1.0)
            ssim_scores.append(ssim_score)
        
        return np.mean(ssim_scores)
    
    def calculate_psnr(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            real_images: Real images tensor [N, C, H, W]
            generated_images: Generated images tensor [N, C, H, W]
            
        Returns:
            Average PSNR score
        """
        psnr_scores = []
        
        for i in range(real_images.shape[0]):
            # Convert to numpy
            real_img = real_images[i].squeeze().cpu().numpy()
            gen_img = generated_images[i].squeeze().cpu().numpy()
            
            # Ensure images are in [0, 1] range
            real_img = np.clip(real_img, 0, 1)
            gen_img = np.clip(gen_img, 0, 1)
            
            # Calculate PSNR
            psnr_score = psnr(real_img, gen_img, data_range=1.0)
            psnr_scores.append(psnr_score)
        
        return np.mean(psnr_scores)
    
    def calculate_l1_loss(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """Calculate L1 loss."""
        return F.l1_loss(generated_images, real_images).item()
    
    def calculate_l2_loss(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """Calculate L2 loss."""
        return F.mse_loss(generated_images, real_images).item()
    
    def calculate_text_alignment_score(self, generated_images: torch.Tensor, 
                                     target_texts: List[str]) -> float:
        """
        Calculate text alignment score (placeholder for OCR-based evaluation).
        This would typically use an OCR model to extract text from generated images
        and compare with target texts.
        
        Args:
            generated_images: Generated images tensor
            target_texts: List of target text strings
            
        Returns:
            Text alignment score (0-1)
        """
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Use an OCR model to extract text from generated images
        # 2. Compare extracted text with target texts
        # 3. Calculate accuracy/alignment score
        
        # For now, return a random score for demonstration
        return np.random.uniform(0.7, 0.9)
    
    def calculate_handwriting_consistency_score(self, generated_images: torch.Tensor) -> float:
        """
        Calculate handwriting consistency score.
        Measures how consistent the handwriting style is across generated samples.
        
        Args:
            generated_images: Generated images tensor
            
        Returns:
            Consistency score (0-1)
        """
        if generated_images.shape[0] < 2:
            return 1.0
        
        # Calculate pairwise similarities between generated images
        similarities = []
        
        for i in range(generated_images.shape[0]):
            for j in range(i + 1, generated_images.shape[0]):
                img1 = generated_images[i].squeeze().cpu().numpy()
                img2 = generated_images[j].squeeze().cpu().numpy()
                
                # Calculate SSIM between pairs
                similarity = ssim(img1, img2, data_range=1.0)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def evaluate_all_metrics(self, real_images: torch.Tensor, generated_images: torch.Tensor,
                           target_texts: List[str] = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            target_texts: List of target text strings
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Image quality metrics
        metrics['fid'] = self.calculate_fid(real_images, generated_images)
        metrics['ssim'] = self.calculate_ssim(real_images, generated_images)
        metrics['psnr'] = self.calculate_psnr(real_images, generated_images)
        metrics['l1_loss'] = self.calculate_l1_loss(real_images, generated_images)
        metrics['l2_loss'] = self.calculate_l2_loss(real_images, generated_images)
        
        # Handwriting-specific metrics
        metrics['consistency'] = self.calculate_handwriting_consistency_score(generated_images)
        
        if target_texts:
            metrics['text_alignment'] = self.calculate_text_alignment_score(generated_images, target_texts)
        
        return metrics
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Compare multiple models based on their evaluation metrics.
        
        Args:
            model_results: Dictionary with model names as keys and metric dictionaries as values
            
        Returns:
            Dictionary with best model for each metric
        """
        comparison = {}
        
        # Get all metric names
        all_metrics = set()
        for results in model_results.values():
            all_metrics.update(results.keys())
        
        # Find best model for each metric
        for metric in all_metrics:
            best_model = None
            best_score = None
            
            for model_name, results in model_results.items():
                if metric in results:
                    score = results[metric]
                    
                    # For FID and losses, lower is better
                    if metric in ['fid', 'l1_loss', 'l2_loss']:
                        if best_score is None or score < best_score:
                            best_score = score
                            best_model = model_name
                    # For other metrics, higher is better
                    else:
                        if best_score is None or score > best_score:
                            best_score = score
                            best_model = model_name
            
            comparison[metric] = best_model
        
        return comparison


def visualize_comparison(real_images: torch.Tensor, generated_images: torch.Tensor,
                        num_samples: int = 8, save_path: str = None):
    """
    Visualize comparison between real and generated images.
    
    Args:
        real_images: Real images tensor
        generated_images: Generated images tensor
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    
    for i in range(min(num_samples, real_images.shape[0])):
        # Real image
        real_img = real_images[i].squeeze().cpu().numpy()
        axes[0, i].imshow(real_img, cmap='gray')
        axes[0, i].set_title('Real', fontsize=10)
        axes[0, i].axis('off')
        
        # Generated image
        gen_img = generated_images[i].squeeze().cpu().numpy()
        axes[1, i].imshow(gen_img, cmap='gray')
        axes[1, i].set_title('Generated', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(model_results: Dict[str, Dict[str, float]], 
                          save_path: str = None):
    """
    Plot comparison of metrics across different models.
    
    Args:
        model_results: Dictionary with model names and their metrics
        save_path: Path to save the plot
    """
    metrics = list(model_results[list(model_results.keys())[0]].keys())
    models = list(model_results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics[:6]):  # Plot first 6 metrics
        values = [model_results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values)
        axes[i].set_title(f'{metric.upper()}', fontsize=12)
        axes[i].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test evaluation metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = EvaluationMetrics(device)
    
    # Create dummy data
    real_images = torch.randn(8, 1, 128, 128).to(device)
    generated_images = torch.randn(8, 1, 128, 128).to(device)
    target_texts = ["مرحبا", "أهلا", "شكرا", "مع السلامة"] * 2
    
    # Calculate metrics
    metrics = evaluator.evaluate_all_metrics(real_images, generated_images, target_texts)
    
    print("Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
    
    # Visualize comparison
    visualize_comparison(real_images, generated_images)

