"""
Classifier model for supervised fine-tuning.

Wraps the MultivarWav2Vec2 encoder with a classification head for:
1. Semiology classification (multi-class)
2. Ictal vs Interictal detection (binary)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import torch
import torch.nn as nn
from model_v5 import MultivarWav2Vec2
from model_config import MODEL_CONFIG


class MultivarWav2Vec2Classifier(nn.Module):
    """
    Classifier based on MultivarWav2Vec2 encoder.
    
    Removes the SwaV self-supervised components and adds a classification head.
    """
    
    def __init__(self, num_classes, config=MODEL_CONFIG, subsample=1, freeze_encoder=False):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes
            config: Model configuration (from model_config.py)
            subsample: Subsampling factor for channels
            freeze_encoder: If True, freeze encoder weights (feature extraction mode)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.config = config
        self.subsample = subsample
        self.freeze_encoder = freeze_encoder
        
        # Create base encoder
        self.encoder = self._create_encoder(config, subsample)
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
        
        # Get embedding dimension from config
        if subsample != 1:
            embedding_dim = int(config.temporal_transformer_hidden * subsample)
        else:
            embedding_dim = config.temporal_transformer_hidden
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        self.embedding_dim = embedding_dim
    
    def _create_encoder(self, config, subsample):
        """Create the encoder without SwaV components."""
        
        # We need to create a custom encoder without projection head and prototypes
        # Let's create a modified version of MultivarWav2Vec2
        class EncoderOnly(nn.Module):
            def __init__(self, config, subsample):
                super().__init__()
                
                if subsample != 1:
                    config = type('obj', (object,), vars(config))()  # Make a copy
                    config.channel_buffer_size = int(config.channel_buffer_size * subsample)
                    config.temporal_transformer_hidden = int(config.temporal_transformer_hidden * subsample)
                
                # Feature encoder (1D convolutions)
                self.ft_enc = nn.ModuleList()
                for i, _ in enumerate(config.ft_enc_dims):
                    if i == 0:
                        self.ft_enc.append(
                            nn.Conv1d(
                                in_channels=1,
                                out_channels=config.ft_enc_dims[i],
                                kernel_size=config.ft_enc_kernel_widths[i],
                                stride=config.ft_enc_strides[i],
                                padding=0,
                                groups=1,
                            )
                        )
                    else:
                        self.ft_enc.append(
                            nn.Conv1d(
                                in_channels=config.ft_enc_dims[i - 1],
                                out_channels=config.ft_enc_dims[i],
                                kernel_size=config.ft_enc_kernel_widths[i],
                                stride=config.ft_enc_strides[i],
                                padding=0,
                                groups=1,
                            )
                        )
                    # Transpose
                    from model_v5 import Transpose
                    self.ft_enc.append(Transpose(1, 2))
                    # LayerNorm
                    self.ft_enc.append(nn.LayerNorm(config.ft_enc_dims[i]))
                    # GELU
                    self.ft_enc.append(nn.GELU())
                    # Transpose back
                    self.ft_enc.append(Transpose(1, 2))
                
                # Adaptive pooling
                self.ft_enc.append(nn.AdaptiveAvgPool1d(config.frames))
                self.ft_enc = nn.Sequential(*self.ft_enc)
                
                # Positional encoding
                from model_v5 import PositionalEncoding
                self.spatiotemporal_pos_encoder = PositionalEncoding(
                    config.ft_enc_dims[-1], config.max_regs, config.frames
                )
                
                # Transformers
                from model_v5 import Transformer
                self.spatial_transformer = Transformer(
                    config.spatial_transformer_hidden,
                    config.spatial_transformer_blocks,
                    config.spatial_transformer_heads,
                    config.spatial_transformer_inner_heads,
                    config.spatial_transformer_mlp_dim
                )
                
                self.temporal_transformer = Transformer(
                    config.temporal_transformer_hidden,
                    config.temporal_transformer_blocks,
                    config.temporal_transformer_heads,
                    config.temporal_transformer_inner_heads,
                    config.temporal_transformer_mlp_dim
                )
                
                # CLS token
                self.class_token = nn.Parameter(torch.randn(1, 1, config.temporal_transformer_hidden))
                
                self.config = config
            
            def forward(self, x, regs):
                from einops import rearrange, repeat
                
                # Feature encoding
                batch_size = x.size(0)
                x = rearrange(x, 'b c n -> (b c) n').unsqueeze(1)
                x = self.ft_enc(x)
                x = rearrange(x, '(b c) d f -> b c d f', b=batch_size)
                x = rearrange(x, 'b c d f -> b c f d')
                
                # Positional encoding
                x = self.spatiotemporal_pos_encoder(x, regs)
                
                # Create mask
                mask = (regs != 0)
                
                # Spatial transformer
                x = rearrange(x, 'b c f d -> b c (f d)')
                x = self.spatial_transformer(x, mask)
                x = rearrange(x, 'b c (f d) -> b c f d', f=self.config.frames)
                
                # Temporal transformer
                x = rearrange(x, 'b c f d -> b f (c d)', f=self.config.frames)
                
                # Add CLS token
                cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=x.size(0))
                x = torch.cat((cls_tokens, x), dim=1)
                
                x = self.temporal_transformer(x)
                
                # Extract CLS token
                x = x[:, 0]
                
                return x
        
        return EncoderOnly(config, subsample)
    
    def _freeze_encoder(self):
        """Freeze encoder parameters for feature extraction."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen (feature extraction mode)")
    
    def forward(self, x, regs):
        """
        Forward pass.
        
        Args:
            x: Input signals (batch, channels, timesteps)
            regs: Region labels (batch, channels)
        
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        # Get embeddings from encoder
        embeddings = self.encoder(x, regs)
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits
    
    def get_embeddings(self, x, regs):
        """
        Get embeddings without classification.
        
        Args:
            x: Input signals (batch, channels, timesteps)
            regs: Region labels (batch, channels)
        
        Returns:
            embeddings: CLS token embeddings (batch, embedding_dim)
        """
        with torch.no_grad():
            embeddings = self.encoder(x, regs)
        return embeddings
    
    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pretrained encoder weights from a self-supervised checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading pretrained weights from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out non-encoder keys (SwaV components)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present (Lightning wrapper)
            if key.startswith('model.'):
                key = key[6:]
            
            # Skip SwaV components
            if 'projection_head' in key or 'prototypes' in key:
                continue
            
            # Map to encoder keys
            if not key.startswith('encoder.'):
                encoder_state_dict[f'encoder.{key}'] = value
            else:
                encoder_state_dict[key] = value
        
        # Load state dict (strict=False to allow missing classifier weights)
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=False)
        
        print(f"✓ Loaded pretrained encoder")
        print(f"  - Missing keys (expected for classifier): {len([k for k in missing_keys if 'classifier' in k])}")
        print(f"  - Unexpected keys: {len(unexpected_keys)}")
        
        return self


if __name__ == "__main__":
    # Test the classifier
    print("Testing MultivarWav2Vec2Classifier...")
    
    # Create model
    model = MultivarWav2Vec2Classifier(num_classes=5, freeze_encoder=False)
    
    # Create dummy input
    batch_size = 2
    channels = 150
    timesteps = 2560
    
    x = torch.randn(batch_size, channels, timesteps)
    regs = torch.randint(0, 41, (batch_size, channels))
    
    # Forward pass
    logits = model(x, regs)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 5)")
    
    # Test embedding extraction
    embeddings = model.get_embeddings(x, regs)
    print(f"✓ Embedding extraction successful")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Expected: ({batch_size}, {model.embedding_dim})")
