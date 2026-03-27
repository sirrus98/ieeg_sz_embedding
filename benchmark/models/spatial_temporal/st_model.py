"""
Spatial-Temporal (ST) Transformer Model for Multi-Channel iEEG Classification.

This model uses a dual-transformer architecture:
1. Spatial transformer: Attends across channels at each time frame, using region embeddings
2. Temporal transformer: Attends across time frames with adaptive channel pooling

Key innovation: Instead of mean pooling channels (bottleneck), uses adaptive pooling
to preserve spatial structure through 20 representative channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conformer.conformer_1d import TransformerEncoder


class ChannelCNNEncoderTemporal(nn.Module):
    """
    Channel-wise 1D CNN encoder that preserves temporal structure.
    
    Instead of pooling to a single vector, outputs a sequence of temporal frames.
    Each channel is processed independently.
    """
    def __init__(self, embed_dim=128, n_frames=40):
        """
        Args:
            embed_dim: Output embedding dimension
            n_frames: Number of temporal frames to output
        """
        super().__init__()
        self.n_frames = n_frames
        self.embed_dim = embed_dim
        
        # 1D CNN layers for temporal feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Adaptive pooling to fixed number of frames
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_frames)
        
        # Project to embedding dimension
        self.fc = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, timesteps] - Single channel time series
            
        Returns:
            [batch, n_frames, embed_dim] - Temporal sequence of embeddings
        """
        # Conv layers with downsampling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adaptive pool to n_frames
        x = self.adaptive_pool(x)  # [batch, 128, n_frames]
        
        # Transpose and project
        x = x.transpose(1, 2)  # [batch, n_frames, 128]
        x = self.fc(x)  # [batch, n_frames, embed_dim]
        
        return x


class SpatialTemporalModel(nn.Module):
    """
    Spatial-Temporal Transformer with adaptive channel pooling.
    
    Architecture:
    1. Channel-wise CNN: Extract temporal features per channel
    2. Spatial Transformer: Attend across channels at each time frame (with region embeddings)
    3. Adaptive Pooling: 150 channels → 20 representative channels (preserves spatial structure)
    4. Projection: Flatten and project to temporal embedding dimension
    5. Temporal Transformer: Attend across time with CLS token
    6. Classification: Binary seizure detection
    """
    
    def __init__(
        self,
        spatial_embed_dim=128,
        temporal_embed_dim=128,
        depth_spatial=8,
        depth_temporal=8,
        n_classes=2,
        max_channels=150,
        n_frames=30,
        n_spatial_tokens=20,
        max_regions=42,
        num_heads=8,
        dropout=0.1
    ):
        """
        Args:
            spatial_embed_dim: Embedding dimension for spatial processing
            temporal_embed_dim: Embedding dimension for temporal processing (now same as spatial)
            depth_spatial: Number of spatial transformer layers
            depth_temporal: Number of temporal transformer layers
            n_classes: Number of output classes
            max_channels: Maximum number of input channels
            n_frames: Number of temporal frames from CNN
            n_spatial_tokens: DEPRECATED - using mean pooling instead
            max_regions: Maximum number of brain regions (for embeddings)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Use same embedding dimension throughout (simplified architecture)
        self.embed_dim = spatial_embed_dim
        self.max_channels = max_channels
        self.n_frames = n_frames
        self.max_regions = max_regions
        
        # Channel-wise CNN encoder (preserves temporal structure)
        self.channel_encoder = ChannelCNNEncoderTemporal(
            embed_dim=self.embed_dim,
            n_frames=n_frames
        )
        
        # Region embeddings for spatial positional encoding
        self.region_embedding = nn.Embedding(max_regions, self.embed_dim)
        
        # Spatial transformer (processes channels at each time frame)
        self.spatial_transformer = TransformerEncoder(
            depth=depth_spatial,
            emb_size=self.embed_dim
        )
        
        # Temporal positional embeddings (learnable)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, n_frames + 1, self.embed_dim)
        )
        
        # Temporal CLS token
        self.temporal_cls_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )
        
        # Temporal transformer (processes time frames)
        self.temporal_transformer = TransformerEncoder(
            depth=depth_temporal,
            emb_size=self.embed_dim
        )
        
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        hidden_dim = self.embed_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x, regs):
        """
        Forward pass.
        
        Args:
            x: [batch, channels, timesteps] - Multi-channel time series
            regs: [batch, channels] - Region IDs for each channel
            
        Returns:
            [batch, n_classes] - Classification logits
        """
        batch, channels, timesteps = x.shape
        
        # === 1. Channel-wise CNN Encoding ===
        # Process each channel independently to get temporal features
        x_reshaped = x.view(batch * channels, 1, timesteps)
        channel_embeds = self.channel_encoder(x_reshaped)
        # Output: [batch*channels, n_frames, spatial_embed_dim]
        
        # Reshape to separate batch and channels
        channel_embeds = channel_embeds.view(
            batch, channels, self.n_frames, self.embed_dim
        )
        # Output: [batch, channels, n_frames, embed_dim]
        
        # === 2. Spatial Transformer ===
        # Process each frame across channels (attend across channels at each time)
        x_spatial = rearrange(channel_embeds, 'b c f d -> (b f) c d')
        # Output: [batch*n_frames, channels, embed_dim]
        
        # Add region-based positional embeddings
        regs_expanded = repeat(regs, 'b c -> (b f) c', f=self.n_frames)
        region_pos = self.region_embedding(regs_expanded)
        # Output: [batch*n_frames, channels, embed_dim]
        
        x_spatial = x_spatial + region_pos
        x_spatial = self.spatial_transformer(x_spatial)
        # Output: [batch*n_frames, channels, embed_dim]
        
        # === 3. Channel Aggregation (Mean Pooling) ===
        # Aggregate channels via mean pooling (standard approach, like Conformer1DDualRegions)
        x_spatial = x_spatial.mean(dim=1)
        # Output: [batch*n_frames, embed_dim]
        
        x_spatial = rearrange(x_spatial, '(b f) d -> b f d', b=batch)
        # Output: [batch, n_frames, embed_dim]
        
        # === 4. Temporal Transformer ===
        # Add CLS token
        cls_tokens = self.temporal_cls_token.expand(batch, -1, -1)
        x_temporal = torch.cat([cls_tokens, x_spatial], dim=1)
        # Output: [batch, n_frames+1, embed_dim]
        
        # Add temporal positional embeddings
        x_temporal = x_temporal + self.temporal_pos_embed[:, :self.n_frames + 1, :]
        
        # Temporal transformer
        x_temporal = self.temporal_transformer(x_temporal)
        # Output: [batch, n_frames+1, embed_dim]
        
        # === 5. Classification ===
        # Extract CLS token
        cls_output = x_temporal[:, 0]
        # Output: [batch, embed_dim]
        
        cls_output = self.norm(cls_output)
        cls_output = self.dropout_layer(cls_output)
        logits = self.classifier(cls_output)
        # Output: [batch, n_classes]
        
        return logits


if __name__ == "__main__":
    # Test the model
    print("Testing SpatialTemporalModel")
    print("=" * 80)
    
    # Create model
    model = SpatialTemporalModel(
        spatial_embed_dim=128,
        temporal_embed_dim=128,  # Same as spatial now (using mean pooling)
        depth_spatial=6,
        depth_temporal=6,
        n_classes=2,
        max_channels=150,
        n_frames=30,
        n_spatial_tokens=20,  # Deprecated parameter (using mean pooling)
        max_regions=42
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    n_channels = 94
    seq_len = 2560
    
    x = torch.randn(batch_size, n_channels, seq_len)
    regs = torch.randint(0, 42, (batch_size, n_channels))
    
    print(f"\nInput shapes:")
    print(f"  Signals: {x.shape}")
    print(f"  Regions: {regs.shape}")
    
    logits = model(x, regs)
    print(f"\nOutput shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 2)")
    
    # Test with padded channels
    x_padded = torch.randn(batch_size, 150, seq_len)
    regs_padded = torch.randint(0, 42, (batch_size, 150))
    regs_padded[:, n_channels:] = 0  # Padded channels have region 0
    
    logits_padded = model(x_padded, regs_padded)
    print(f"\nWith padding (150 channels): {logits_padded.shape}")
    
    print("\n✓ SpatialTemporalModel working correctly!")
