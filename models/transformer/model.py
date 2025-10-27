"""
Transformer-based Generator for Arabic Handwriting Generation.
Implements Vision Transformer (ViT) with autoregressive decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TextEncoder(nn.Module):
    """Text encoder for converting Arabic text to embeddings."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, max_length: int = 50):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        
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
        char_embeds = char_embeds.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        char_embeds = self.pos_encoding(char_embeds)
        char_embeds = char_embeds.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Transformer encoding
        encoded = self.transformer(char_embeds)  # [batch_size, seq_len, embed_dim]
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # [batch_size, embed_dim]
        
        # Project to hidden dimension
        output = self.output_proj(pooled)  # [batch_size, hidden_dim]
        
        return output


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer."""
    
    def __init__(self, img_size: int = 128, patch_size: int = 16, 
                 in_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.projection(x)  # [batch_size, embed_dim, n_patches^0.5, n_patches^0.5]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x


class TransformerGenerator(nn.Module):
    """Transformer-based generator for handwriting generation."""
    
    def __init__(self, vocab_size: int = 1000, img_size: int = 128, patch_size: int = 16,
                 embed_dim: int = 768, text_dim: int = 512, num_layers: int = 12,
                 num_heads: int = 12, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=text_dim
        )
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim
        )
        
        # Positional encoding for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size)
        
    def forward(self, text_tokens: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate image from text using transformer.
        
        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            noise: Optional noise for generation [batch_size, n_patches, embed_dim]
            
        Returns:
            Generated image [batch_size, 1, img_size, img_size]
        """
        batch_size = text_tokens.size(0)
        
        # Encode text
        text_embedding = self.text_encoder(text_tokens)  # [batch_size, text_dim]
        
        # Project text to patch dimension
        text_patches = text_embedding.unsqueeze(1).expand(-1, self.n_patches, -1)  # [batch_size, n_patches, embed_dim]
        text_patches = F.linear(text_patches, self.patch_embedding.projection.weight.view(self.embed_dim, -1).T)
        
        # Add noise or use text patches as base
        if noise is not None:
            patch_embeds = noise
        else:
            # Initialize with text-conditioned patches
            patch_embeds = text_patches
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        patch_embeds = torch.cat([cls_tokens, patch_embeds], dim=1)  # [batch_size, n_patches + 1, embed_dim]
        
        # Add positional encoding
        patch_embeds = patch_embeds + self.pos_embedding
        
        # Transformer encoding
        encoded = self.transformer(patch_embeds)  # [batch_size, n_patches + 1, embed_dim]
        
        # Remove CLS token
        patch_outputs = encoded[:, 1:, :]  # [batch_size, n_patches, embed_dim]
        
        # Project to patch values
        patch_values = self.output_proj(patch_outputs)  # [batch_size, n_patches, patch_size^2]
        
        # Reshape to image
        patches_per_side = int(math.sqrt(self.n_patches))
        patch_values = patch_values.view(batch_size, patches_per_side, patches_per_side, 
                                       self.patch_size, self.patch_size)
        
        # Rearrange patches to form image
        image = rearrange(patch_values, 'b h w p1 p2 -> b (h p1) (w p2)')
        image = image.unsqueeze(1)  # Add channel dimension
        
        return image
    
    def generate(self, text_tokens: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate images from text.
        
        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            num_samples: Number of samples to generate per text
            
        Returns:
            Generated images [batch_size * num_samples, 1, img_size, img_size]
        """
        self.eval()
        with torch.no_grad():
            batch_size = text_tokens.size(0)
            
            # Generate multiple samples
            all_generated = []
            for _ in range(num_samples):
                # Sample noise for patches
                noise = torch.randn(batch_size, self.n_patches, self.embed_dim, 
                                  device=text_tokens.device)
                
                generated = self.forward(text_tokens, noise)
                all_generated.append(generated)
            
            # Concatenate all samples
            generated = torch.cat(all_generated, dim=0)
            
        return generated


class TransformerDiscriminator(nn.Module):
    """Transformer-based discriminator."""
    
    def __init__(self, img_size: int = 128, patch_size: int = 16, 
                 embed_dim: int = 768, text_dim: int = 512, 
                 num_layers: int = 6, num_heads: int = 12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=1000,  # Same as generator
            embed_dim=embed_dim,
            hidden_dim=text_dim
        )
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Classify image-text pair as real or fake.
        
        Args:
            image: Input image [batch_size, 1, img_size, img_size]
            text_tokens: Tokenized text [batch_size, seq_len]
            
        Returns:
            Classification score [batch_size, 1]
        """
        batch_size = image.size(0)
        
        # Encode text
        text_embedding = self.text_encoder(text_tokens)  # [batch_size, text_dim]
        
        # Patch embedding
        patch_embeds = self.patch_embedding(image)  # [batch_size, n_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_embeds = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # Add positional encoding
        patch_embeds = patch_embeds + self.pos_embedding
        
        # Transformer encoding
        encoded = self.transformer(patch_embeds)
        
        # Use CLS token for classification
        cls_output = encoded[:, 0, :]  # [batch_size, embed_dim]
        
        # Classify
        score = self.classifier(cls_output)  # [batch_size, 1]
        
        return score


class TransformerGAN(nn.Module):
    """Transformer-based GAN for Arabic Handwriting Generation."""
    
    def __init__(self, vocab_size: int = 1000, img_size: int = 128, patch_size: int = 16,
                 embed_dim: int = 768, text_dim: int = 512, num_layers: int = 12,
                 num_heads: int = 12, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Generator
        self.generator = TransformerGenerator(
            vocab_size=vocab_size,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            text_dim=text_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Discriminator
        self.discriminator = TransformerDiscriminator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            text_dim=text_dim,
            num_layers=num_layers // 2,  # Smaller discriminator
            num_heads=num_heads
        )
        
    def forward(self, text_tokens: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate image from text."""
        return self.generator(text_tokens, noise)
    
    def discriminate(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Discriminate between real and fake images."""
        return self.discriminator(image, text_tokens)


class TransformerTrainer:
    """Trainer class for Transformer-based GAN."""
    
    def __init__(self, model: TransformerGAN, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.g_optimizer = torch.optim.AdamW(model.generator.parameters(), lr=1e-4, weight_decay=0.01)
        self.d_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=1e-4, weight_decay=0.01)
        
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
        fake_images = self.model(text_tokens)
        fake_output = self.model.discriminate(fake_images.detach(), text_tokens)
        d_loss_fake = self.criterion(fake_output.squeeze(), fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        fake_images = self.model(text_tokens)
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
            fake_images = self.model(text_tokens)
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
        text_tokens = self._tokenize_texts(texts).to(self.device)
        generated = self.model.generator.generate(text_tokens, num_samples)
        return generated


if __name__ == "__main__":
    # Test Transformer model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = TransformerGAN(
        vocab_size=1000,
        img_size=128,
        patch_size=16,
        embed_dim=768,
        text_dim=512,
        num_layers=12,
        num_heads=12
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    texts = ["مرحبا", "أهلا", "شكرا", "مع السلامة"]
    text_tokens = model.generator._tokenize_texts(texts)
    
    generated = model(text_tokens)
    print(f"Generated shape: {generated.shape}")
    
    # Test discriminator
    real_images = torch.randn(batch_size, 1, 128, 128)
    scores = model.discriminate(real_images, text_tokens)
    print(f"Discriminator scores shape: {scores.shape}")
    
    # Test trainer
    trainer = TransformerTrainer(model, device)
    print("Transformer model and trainer created successfully!")

