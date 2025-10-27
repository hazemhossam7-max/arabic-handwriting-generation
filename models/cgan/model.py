"""
Conditional Generative Adversarial Network (cGAN) for Arabic Handwriting Generation.
Implements Pix2Pix-style architecture with text conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class TextEncoder(nn.Module):
    """Text encoder for converting Arabic text to embeddings."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, max_length: int = 50):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_length, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text tokens to embeddings.
        
        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            
        Returns:
            Text embeddings [batch_size, hidden_dim]
        """
        batch_size, seq_len = text_tokens.shape
        
        # Character embeddings
        char_embeds = self.char_embedding(text_tokens)  # [batch_size, seq_len, embed_dim]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        char_embeds = char_embeds + pos_enc
        
        # Transformer encoding
        encoded = self.transformer(char_embeds)  # [batch_size, seq_len, embed_dim]
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # [batch_size, embed_dim]
        
        # Project to hidden dimension
        output = self.output_proj(pooled)  # [batch_size, hidden_dim]
        
        return output


class CGANGenerator(nn.Module):
    """Generator network for cGAN (Pix2Pix-style)."""
    
    def __init__(self, noise_dim: int = 100, text_dim: int = 256, 
                 image_channels: int = 1, image_size: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.text_dim = text_dim
        self.image_size = image_size
        
        # Calculate the size after initial projection
        initial_size = image_size // 8  # After 3 downsampling layers
        
        # Initial projection
        self.fc = nn.Linear(noise_dim + text_dim, 512 * initial_size * initial_size)
        
        # Generator network (U-Net style)
        self.generator = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, noise: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate image from noise and text.
        
        Args:
            noise: Random noise [batch_size, noise_dim]
            text_embedding: Text embedding [batch_size, text_dim]
            
        Returns:
            Generated image [batch_size, channels, height, width]
        """
        # Concatenate noise and text
        combined = torch.cat([noise, text_embedding], dim=1)  # [batch_size, noise_dim + text_dim]
        
        # Project to feature space
        features = self.fc(combined)  # [batch_size, 512 * initial_size * initial_size]
        features = features.view(-1, 512, self.image_size // 8, self.image_size // 8)
        
        # Generate image
        image = self.generator(features)  # [batch_size, channels, height, width]
        
        return image


class CGANDiscriminator(nn.Module):
    """Discriminator network for cGAN."""
    
    def __init__(self, image_channels: int = 1, text_dim: int = 256):
        super().__init__()
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Text processing
        self.text_proj = nn.Linear(text_dim, 512 * 4 * 4)
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 * 2, 1),  # 2 because we concatenate image and text features
            nn.Sigmoid()
        )
        
    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify image-text pair as real or fake.
        
        Args:
            image: Input image [batch_size, channels, height, width]
            text_embedding: Text embedding [batch_size, text_dim]
            
        Returns:
            Classification score [batch_size, 1]
        """
        # Encode image
        img_features = self.image_encoder(image)  # [batch_size, 512, 4, 4]
        img_flat = img_features.view(img_features.size(0), -1)  # [batch_size, 512*4*4]
        
        # Process text
        text_features = self.text_proj(text_embedding)  # [batch_size, 512*4*4]
        
        # Concatenate image and text features
        combined = torch.cat([img_flat, text_features], dim=1)  # [batch_size, 512*4*4*2]
        
        # Classify
        score = self.classifier(combined)  # [batch_size, 1]
        
        return score


class CGAN(nn.Module):
    """Conditional GAN for Arabic Handwriting Generation."""
    
    def __init__(self, vocab_size: int = 1000, noise_dim: int = 100, 
                 image_channels: int = 1, text_embed_dim: int = 256, 
                 text_hidden_dim: int = 512, image_size: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.image_size = image_size
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim
        )
        
        # Generator
        self.generator = CGANGenerator(
            noise_dim=noise_dim,
            text_dim=text_hidden_dim,
            image_channels=image_channels,
            image_size=image_size
        )
        
        # Discriminator
        self.discriminator = CGANDiscriminator(
            image_channels=image_channels,
            text_dim=text_hidden_dim
        )
        
    def forward(self, text_tokens: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate image from text.
        
        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            noise: Optional noise vector [batch_size, noise_dim]
            
        Returns:
            Generated image [batch_size, channels, height, width]
        """
        # Encode text
        text_embedding = self.text_encoder(text_tokens)  # [batch_size, text_hidden_dim]
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(text_embedding.size(0), self.noise_dim, 
                              device=text_embedding.device)
        
        # Generate image
        generated_image = self.generator(noise, text_embedding)
        
        return generated_image
    
    def discriminate(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between real and fake images.
        
        Args:
            image: Input image [batch_size, channels, height, width]
            text_tokens: Tokenized text [batch_size, seq_len]
            
        Returns:
            Classification score [batch_size, 1]
        """
        # Encode text
        text_embedding = self.text_encoder(text_tokens)
        
        # Discriminate
        score = self.discriminator(image, text_embedding)
        
        return score
    
    def _tokenize_texts(self, texts: list) -> torch.Tensor:
        """Simple tokenization (replace with proper Arabic tokenizer)."""
        # This is a placeholder - in practice, use proper Arabic text tokenization
        max_length = 50
        vocab_size = 1000
        
        tokenized = []
        for text in texts:
            # Simple character-based tokenization
            tokens = [ord(c) % vocab_size for c in text[:max_length]]
            # Pad to max_length
            tokens += [0] * (max_length - len(tokens))
            tokenized.append(tokens)
        
        return torch.tensor(tokenized, dtype=torch.long)


class CGANTrainer:
    """Trainer class for cGAN."""
    
    def __init__(self, model: CGAN, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Labels
        self.real_label = 1.0
        self.fake_label = 0.0
        
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        
        images = batch['images'].to(self.device)
        texts = batch['processed_texts']
        batch_size = images.size(0)
        
        # Tokenize texts
        text_tokens = self._tokenize_texts(texts).to(self.device)
        
        # Create labels
        real_labels = torch.full((batch_size,), self.real_label, device=self.device)
        fake_labels = torch.full((batch_size,), self.fake_label, device=self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_output = self.model.discriminate(images, text_tokens)
        d_loss_real = self.criterion(real_output.squeeze(), real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.model.noise_dim, device=self.device)
        fake_images = self.model.generator(noise, self.model.text_encoder(text_tokens))
        fake_output = self.model.discriminate(fake_images.detach(), text_tokens)
        d_loss_fake = self.criterion(fake_output.squeeze(), fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, self.model.noise_dim, device=self.device)
        fake_images = self.model.generator(noise, self.model.text_encoder(text_tokens))
        fake_output = self.model.discriminate(fake_images, text_tokens)
        
        # Generator loss
        g_loss = self.criterion(fake_output.squeeze(), real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item()
        }
    
    def _tokenize_texts(self, texts: list) -> torch.Tensor:
        """Simple tokenization (replace with proper Arabic tokenizer)."""
        max_length = 50
        vocab_size = 1000
        
        tokenized = []
        for text in texts:
            # Simple character-based tokenization
            tokens = [ord(c) % vocab_size for c in text[:max_length]]
            # Pad to max_length
            tokens += [0] * (max_length - len(tokens))
            tokenized.append(tokens)
        
        return torch.tensor(tokenized, dtype=torch.long)
    
    def validate_step(self, batch: dict) -> dict:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            images = batch['images'].to(self.device)
            texts = batch['processed_texts']
            batch_size = images.size(0)
            
            text_tokens = self._tokenize_texts(texts).to(self.device)
            
            # Create labels
            real_labels = torch.full((batch_size,), self.real_label, device=self.device)
            fake_labels = torch.full((batch_size,), self.fake_label, device=self.device)
            
            # Real images
            real_output = self.model.discriminate(images, text_tokens)
            d_loss_real = self.criterion(real_output.squeeze(), real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.model.noise_dim, device=self.device)
            fake_images = self.model.generator(noise, self.model.text_encoder(text_tokens))
            fake_output = self.model.discriminate(fake_images, text_tokens)
            d_loss_fake = self.criterion(fake_output.squeeze(), fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            g_loss = self.criterion(fake_output.squeeze(), real_labels)
            
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item()
        }
    
    def generate_samples(self, texts: list, num_samples: int = 4) -> torch.Tensor:
        """Generate samples for given texts."""
        self.model.eval()
        
        with torch.no_grad():
            text_tokens = self._tokenize_texts(texts).to(self.device)
            
            # Generate multiple samples per text
            all_generated = []
            for _ in range(num_samples):
                noise = torch.randn(len(texts), self.model.noise_dim, device=self.device)
                generated = self.model.generator(noise, self.model.text_encoder(text_tokens))
                all_generated.append(generated)
            
            # Concatenate all samples
            generated = torch.cat(all_generated, dim=0)
            
        return generated


if __name__ == "__main__":
    # Test cGAN model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = CGAN(
        vocab_size=1000,
        noise_dim=100,
        image_channels=1,
        text_embed_dim=256,
        text_hidden_dim=512,
        image_size=128
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    texts = ["مرحبا", "أهلا", "شكرا", "مع السلامة"]
    text_tokens = model._tokenize_texts(texts)
    
    generated = model(text_tokens)
    print(f"Generated shape: {generated.shape}")
    
    # Test discriminator
    real_images = torch.randn(batch_size, 1, 128, 128)
    scores = model.discriminate(real_images, text_tokens)
    print(f"Discriminator scores shape: {scores.shape}")
    
    # Test trainer
    trainer = CGANTrainer(model, device)
    print("cGAN model and trainer created successfully!")

