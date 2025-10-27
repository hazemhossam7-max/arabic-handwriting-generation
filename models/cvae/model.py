"""
Conditional Variational Autoencoder (CVAE) for Arabic Handwriting Generation.
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


class CVAEEncoder(nn.Module):
    """Encoder network for CVAE."""
    
    def __init__(self, image_channels: int = 1, text_dim: int = 256, 
                 latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 8x8 -> 4x4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 4 * 4 + text_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4 + text_dim, latent_dim)
        
    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image and text to latent space.
        
        Args:
            image: Input image [batch_size, channels, height, width]
            text_embedding: Text embedding [batch_size, text_dim]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode image
        img_features = self.image_encoder(image)  # [batch_size, 512, 4, 4]
        img_flat = self.flatten(img_features)  # [batch_size, 512*4*4]
        
        # Concatenate with text embedding
        combined = torch.cat([img_flat, text_embedding], dim=1)  # [batch_size, 512*4*4 + text_dim]
        
        # Project to latent space
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar


class CVAEDecoder(nn.Module):
    """Decoder network for CVAE."""
    
    def __init__(self, latent_dim: int = 128, text_dim: int = 256, 
                 image_channels: int = 1, hidden_dim: int = 512):
        super().__init__()
        
        # Project latent + text to feature space
        self.fc = nn.Linear(latent_dim + text_dim, 512 * 4 * 4)
        
        # Image decoder
        self.image_decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
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
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector and text to image.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            text_embedding: Text embedding [batch_size, text_dim]
            
        Returns:
            Generated image [batch_size, channels, height, width]
        """
        # Concatenate latent and text
        combined = torch.cat([z, text_embedding], dim=1)  # [batch_size, latent_dim + text_dim]
        
        # Project to feature space
        features = self.fc(combined)  # [batch_size, 512*4*4]
        features = features.view(-1, 512, 4, 4)  # [batch_size, 512, 4, 4]
        
        # Decode to image
        image = self.image_decoder(features)  # [batch_size, channels, height, width]
        
        return image


class CVAE(nn.Module):
    """Conditional Variational Autoencoder for Arabic Handwriting Generation."""
    
    def __init__(self, vocab_size: int = 1000, image_channels: int = 1, 
                 latent_dim: int = 128, text_embed_dim: int = 256, 
                 text_hidden_dim: int = 512, image_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim
        )
        
        # CVAE encoder
        self.encoder = CVAEEncoder(
            image_channels=image_channels,
            text_dim=text_hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=512
        )
        
        # CVAE decoder
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim,
            text_dim=text_hidden_dim,
            image_channels=image_channels,
            hidden_dim=512
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
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
    
    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of CVAE.
        
        Args:
            image: Input image [batch_size, channels, height, width]
            text_tokens: Tokenized text [batch_size, seq_len]
            
        Returns:
            reconstructed_image: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode text
        text_embedding = self.text_encoder(text_tokens)  # [batch_size, text_hidden_dim]
        
        # Encode image and text to latent space
        mu, logvar = self.encoder(image, text_embedding)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode to image
        reconstructed_image = self.decoder(z, text_embedding)
        
        return reconstructed_image, mu, logvar
    
    def generate(self, text_tokens: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate images from text.
        
        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            num_samples: Number of samples to generate per text
            
        Returns:
            Generated images [batch_size * num_samples, channels, height, width]
        """
        self.eval()
        with torch.no_grad():
            # Encode text
            text_embedding = self.text_encoder(text_tokens)  # [batch_size, text_hidden_dim]
            
            # Repeat text embedding for multiple samples
            if num_samples > 1:
                text_embedding = text_embedding.repeat_interleave(num_samples, dim=0)
            
            # Sample from prior (standard normal)
            z = torch.randn(text_embedding.shape[0], self.latent_dim, device=text_embedding.device)
            
            # Decode to image
            generated_images = self.decoder(z, text_embedding)
            
        return generated_images
    
    def loss_function(self, reconstructed_image: torch.Tensor, original_image: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        Calculate CVAE loss.
        
        Args:
            reconstructed_image: Reconstructed image
            original_image: Original image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence term
            
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed_image, original_image, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        loss_dict = {
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class CVAETrainer:
    """Trainer class for CVAE."""
    
    def __init__(self, model: CVAE, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        
        images = batch['images'].to(self.device)
        texts = batch['processed_texts']
        
        # Tokenize texts (simplified - in practice, use proper tokenization)
        text_tokens = self._tokenize_texts(texts).to(self.device)
        
        # Forward pass
        reconstructed, mu, logvar = self.model(images, text_tokens)
        
        # Calculate loss
        loss, loss_dict = self.model.loss_function(reconstructed, images, mu, logvar)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss_dict
    
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
    
    def validate_step(self, batch: dict) -> dict:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            images = batch['images'].to(self.device)
            texts = batch['processed_texts']
            
            text_tokens = self._tokenize_texts(texts).to(self.device)
            
            reconstructed, mu, logvar = self.model(images, text_tokens)
            loss, loss_dict = self.model.loss_function(reconstructed, images, mu, logvar)
            
        return loss_dict
    
    def generate_samples(self, texts: list, num_samples: int = 4) -> torch.Tensor:
        """Generate samples for given texts."""
        text_tokens = self._tokenize_texts(texts).to(self.device)
        generated = self.model.generate(text_tokens, num_samples)
        return generated


if __name__ == "__main__":
    # Test CVAE model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = CVAE(
        vocab_size=1000,
        image_channels=1,
        latent_dim=128,
        text_embed_dim=256,
        text_hidden_dim=512,
        image_size=128
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 1, 128, 128)
    texts = ["مرحبا", "أهلا", "شكرا", "مع السلامة"]
    
    reconstructed, mu, logvar = model(images, model._tokenize_texts(texts))
    
    print(f"Input shape: {images.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test generation
    generated = model.generate(model._tokenize_texts(texts[:2]), num_samples=2)
    print(f"Generated shape: {generated.shape}")
    
    # Test trainer
    trainer = CVAETrainer(model, device)
    print("CVAE model and trainer created successfully!")

