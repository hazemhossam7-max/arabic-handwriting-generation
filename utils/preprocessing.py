"""
Data preprocessing utilities for Arabic handwriting generation.
Handles loading, cleaning, and augmentation of Arabic handwriting datasets.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ArabicTextProcessor:
    """Handles Arabic text preprocessing and reshaping."""
    
    def __init__(self):
        self.reshaper = arabic_reshaper.ArabicReshaper()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess Arabic text for proper display and processing.
        
        Args:
            text: Raw Arabic text
            
        Returns:
            Processed Arabic text
        """
        # Remove extra whitespace
        text = text.strip()
        
        # Reshape Arabic text for proper display
        reshaped_text = self.reshaper.reshape(text)
        
        # Apply bidirectional algorithm for correct text direction
        display_text = get_display(reshaped_text)
        
        return display_text
    
    def extract_characters(self, text: str) -> List[str]:
        """
        Extract individual Arabic characters from text.
        
        Args:
            text: Arabic text
            
        Returns:
            List of individual characters
        """
        # Remove diacritics and extract base characters
        chars = []
        for char in text:
            if char.strip() and char not in [' ', '\n', '\t']:
                chars.append(char)
        return chars


class ImagePreprocessor:
    """Handles image preprocessing for Arabic handwriting."""
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for training.
        
        Args:
            image: Input image array
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image
        """
        if image is None:
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using various techniques.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to numpy
        enhanced = np.array(pil_image) / 255.0
        
        return enhanced


class DataAugmentation:
    """Handles data augmentation for Arabic handwriting."""
    
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(image=image)['image']
        
        return augmented


class ArabicHandwritingDataset:
    """Dataset class for Arabic handwriting data."""
    
    def __init__(self, data_dir: str, target_size: Tuple[int, int] = (128, 128)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.text_processor = ArabicTextProcessor()
        self.image_preprocessor = ImagePreprocessor(target_size)
        self.augmentation = DataAugmentation()
        
        self.samples = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from directory structure."""
        # Expected structure: data_dir/images/ and data_dir/texts/
        images_dir = os.path.join(self.data_dir, 'images')
        texts_dir = os.path.join(self.data_dir, 'texts')
        
        if not os.path.exists(images_dir) or not os.path.exists(texts_dir):
            print(f"Warning: Expected directories not found. Creating placeholder structure.")
            return
        
        # Load image-text pairs
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, filename)
                text_path = os.path.join(texts_dir, filename.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))
                
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    self.samples.append({
                        'image_path': image_path,
                        'text': text,
                        'processed_text': self.text_processor.preprocess_text(text)
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = self.image_preprocessor.load_image(sample['image_path'])
        image = self.image_preprocessor.preprocess_image(image)
        
        if image is None:
            # Return a blank image if loading fails
            image = np.zeros(self.target_size, dtype=np.float32)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image_tensor,
            'text': sample['text'],
            'processed_text': sample['processed_text']
        }
    
    def get_augmented_sample(self, idx):
        """Get an augmented sample."""
        sample = self.__getitem__(idx)
        
        # Apply augmentation
        image_np = sample['image'].squeeze().numpy()
        augmented_image = self.augmentation.augment_image(image_np)
        
        # Convert back to tensor
        augmented_tensor = torch.from_numpy(augmented_image).float()
        if len(augmented_tensor.shape) == 2:
            augmented_tensor = augmented_tensor.unsqueeze(0)
        
        sample['image'] = augmented_tensor
        return sample


class DataLoader:
    """Data loader for training and evaluation."""
    
    def __init__(self, dataset: ArabicHandwritingDataset, batch_size: int = 32, 
                 shuffle: bool = True, augmentation: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        
    def create_dataloader(self):
        """Create PyTorch DataLoader."""
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            texts = [item['text'] for item in batch]
            processed_texts = [item['processed_text'] for item in batch]
            
            return {
                'images': images,
                'texts': texts,
                'processed_texts': processed_texts
            }
        
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
            num_workers=2
        )
    
    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.1):
        """Split dataset into train, validation, and test sets."""
        indices = list(range(len(self.dataset)))
        
        # Split into train and temp
        train_indices, temp_indices = train_test_split(
            indices, test_size=test_size + val_size, random_state=42
        )
        
        # Split temp into validation and test
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=test_size/(test_size + val_size), random_state=42
        )
        
        return train_indices, val_indices, test_indices


def visualize_samples(dataset: ArabicHandwritingDataset, num_samples: int = 8):
    """Visualize sample images and texts."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Display image
        image = sample['image'].squeeze().numpy()
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Text: {sample['text'][:20]}...", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_placeholder_dataset(output_dir: str, num_samples: int = 100):
    """Create a placeholder dataset for testing."""
    images_dir = os.path.join(output_dir, 'images')
    texts_dir = os.path.join(output_dir, 'texts')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    # Sample Arabic texts
    sample_texts = [
        "مرحبا بالعالم",
        "أهلا وسهلا",
        "كيف حالك",
        "شكرا لك",
        "مع السلامة",
        "صباح الخير",
        "مساء الخير",
        "أحبك",
        "السلام عليكم",
        "بارك الله فيك"
    ]
    
    for i in range(num_samples):
        # Create random image
        image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        # Add some structure to make it look more like handwriting
        cv2.circle(image, (64, 64), 30, 0, -1)
        
        # Save image
        image_path = os.path.join(images_dir, f'sample_{i:04d}.png')
        cv2.imwrite(image_path, image)
        
        # Save corresponding text
        text = sample_texts[i % len(sample_texts)]
        text_path = os.path.join(texts_dir, f'sample_{i:04d}.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == "__main__":
    # Create placeholder dataset for testing
    create_placeholder_dataset('data/raw', 50)
    
    # Test the dataset
    dataset = ArabicHandwritingDataset('data/raw')
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Visualize samples
    visualize_samples(dataset, 8)
    
    # Test data loader
    data_loader = DataLoader(dataset, batch_size=4)
    loader = data_loader.create_dataloader()
    
    # Test a batch
    for batch in loader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch texts: {batch['texts']}")
        break
