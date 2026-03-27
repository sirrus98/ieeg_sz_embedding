"""
Conformer1D - ViT-style Multi-Channel Transformer for iEEG Seizure Detection.

This version uses:
- Channel-wise 1D CNN encoders (parallelized batch processing)
- ViT-style transformer with learnable class token
- Positional embeddings for channel ordering
- Self-contained architecture (no separate wrapper aggregation needed)

Key optimization: All channels are processed in parallel using batched operations,
eliminating the inefficient for-loop in the original notebook implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce


# ===== Transformer Components (from original conformer.py) =====

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=2,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


# ===== NEW: Channel-wise 1D CNN Encoder =====

class ChannelCNNEncoder(nn.Module):
    """
    1D CNN to encode each channel independently.
    
    Processes univariate time series [batch, 1, timesteps] and outputs 
    a fixed-size embedding [batch, embed_dim].
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Conv layers with batch norm and pooling
        self.conv1 = nn.Conv1d(1, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        
        # Adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to embedding dimension
        self.fc = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, timesteps] - univariate time series
        
        Returns:
            [batch, embed_dim] - channel embedding
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)  # [batch, 128, 1]
        x = x.squeeze(-1)  # [batch, 128]
        x = self.fc(x)  # [batch, embed_dim]
        return x


# ===== NEW: Multi-Channel ViT-style Conformer =====

class Conformer1D(nn.Module):
    """
    Multi-channel transformer classifier using ViT-style architecture.
    
    Architecture:
    1. Channel-wise CNN encoder (parallelized across all channels)
    2. Learnable class token (ViT-style) - replaces mean/max pooling
    3. Positional embeddings for channel ordering
    4. Transformer encoder
    5. Classification head on class token output
    
    Key features:
    - Processes all channels in parallel (no for loops)
    - Handles variable channel counts via max_channels parameter
    - Self-contained (no separate wrapper needed)
    - Class token learns optimal channel aggregation (no manual mean/max pooling)
    
    Note on aggregation:
    Unlike previous architectures that used simple mean or max pooling across
    channel features, this ViT-style approach uses a learnable class token that
    attends to all channel embeddings through the transformer. This allows the
    model to learn which channels are important and how to combine them optimally
    for the classification task.
    """
    
    def __init__(self, emb_size=128, depth=4, num_heads=8, 
                 n_classes=2, dropout=0.1, max_channels=150):
        """
        Args:
            emb_size: Embedding dimension for channel encodings
            depth: Number of transformer encoder layers
            num_heads: Number of attention heads
            n_classes: Number of output classes
            dropout: Dropout rate
            max_channels: Maximum number of channels to handle
        """
        super().__init__()
        self.max_channels = max_channels
        self.embed_dim = emb_size
        
        # Per-channel CNN encoder
        self.channel_encoder = ChannelCNNEncoder(emb_size)
        
        # Class token (learnable, ViT-style)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        # Positional encoding for channels + class token
        self.pos_embed = nn.Parameter(torch.randn(1, max_channels + 1, emb_size))
        
        # Transformer encoder
        # Use scaled-down parameters for efficiency
        forward_expansion = 4 if emb_size >= 64 else 2
        self.transformer = TransformerEncoder(depth, emb_size)
        
        # Classification head
        hidden_dim = max(32, emb_size * 2)
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x):
        """
        Forward pass with parallelized channel processing.
        
        Args:
            x: [batch, n_channels, timesteps] - multi-channel time series
        
        Returns:
            logits: [batch, n_classes] - classification logits
        """
        batch_size, n_channels, timesteps = x.shape
        
        # OPTIMIZATION: Reshape to process all channels in parallel
        # Instead of looping over channels, we batch them together
        x_reshaped = x.view(batch_size * n_channels, 1, timesteps)
        
        # Encode all channels at once (parallel processing on GPU)
        channel_embeds = self.channel_encoder(x_reshaped)  # [batch*n_channels, embed_dim]
        
        # Reshape back to separate batch and channel dimensions
        channel_embeds = channel_embeds.view(batch_size, n_channels, self.embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        tokens = torch.cat([cls_tokens, channel_embeds], dim=1)  # [batch, n_channels+1, embed_dim]
        
        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :n_channels+1, :]
        
        # Transformer encoder
        tokens = self.transformer(tokens)  # [batch, n_channels+1, embed_dim]
        
        # Extract class token
        cls_output = tokens[:, 0]  # [batch, embed_dim]
        
        # Classification
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch, n_classes]
        
        return logits
    
    def get_embeddings(self, x):
        """
        Extract embeddings (class token output after norm, before classification).
        
        Args:
            x: [batch, n_channels, timesteps] - multi-channel time series
        
        Returns:
            embeddings: [batch, embed_dim] - class token embeddings
        """
        batch_size, n_channels, timesteps = x.shape
        
        # Encode all channels
        x_reshaped = x.view(batch_size * n_channels, 1, timesteps)
        channel_embeds = self.channel_encoder(x_reshaped)
        channel_embeds = channel_embeds.view(batch_size, n_channels, self.embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, channel_embeds], dim=1)
        
        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :n_channels+1, :]
        
        # Transformer encoder
        tokens = self.transformer(tokens)
        
        # Extract class token and normalize (but don't classify)
        cls_output = tokens[:, 0]
        cls_output = self.norm(cls_output)
        # Note: We include dropout during training for consistency with forward pass
        if self.training:
            cls_output = self.dropout(cls_output)
        
        return cls_output


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing Conformer1D Model (ViT-style)")
    print("="*60)
    
    model = Conformer1D(emb_size=128, depth=4, num_heads=8, n_classes=2, max_channels=94)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print()
    
    # Test with multi-channel input (10 seconds @ 256Hz)
    batch_size = 4
    n_channels = 94
    timesteps = 2560
    
    x = torch.randn(batch_size, n_channels, timesteps)
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    
    # Test probabilities
    probs = F.softmax(logits, dim=1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    print()
    
    # Test with variable channels
    x_var = torch.randn(batch_size, 50, timesteps)
    logits_var = model(x_var)
    print(f"Variable channels (50) output: {logits_var.shape}")
    print()
    
    print("✓ Conformer1D working correctly!")
    print("✓ All channels processed in parallel (no for loops)")
