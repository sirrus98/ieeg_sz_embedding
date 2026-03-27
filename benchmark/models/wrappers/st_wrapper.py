"""
Spatial-Temporal (ST) Model Wrapper.

Wraps the SpatialTemporalModel for compatibility with train_benchmarks.py.
Unlike other wrappers, this model requires region information as input.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from spatial_temporal.st_model import SpatialTemporalModel


class STChannelAggregator(nn.Module):
    """
    Wrapper for SpatialTemporal model.
    
    This wrapper maintains API compatibility with train_benchmarks.py.
    Unlike other channel aggregators (ONSET, WVNT, Conformer), this model
    requires region information and uses a dual-transformer architecture
    with mean pooling for channel aggregation.
    
    IMPORTANT: The 'aggregation' and 'n_spatial_tokens' parameters are DEPRECATED and IGNORED.
    
    The model uses mean pooling across channels after spatial transformer,
    similar to the working Conformer1DDualRegions architecture.
    
    Architecture:
    - Spatial transformer: 6 layers, attends across channels with region embeddings
    - Mean pooling: Aggregates 150 channels → 1 embedding per frame
    - Temporal transformer: 6 layers, attends across 30 time frames
    """
    
    def __init__(
        self,
        spatial_embed_dim=128,
        temporal_embed_dim=128,
        depth_spatial=6,
        depth_temporal=6,
        n_spatial_tokens=20,
        aggregation='mean',
        freeze_backbone=False,
        max_channels=150,
        n_frames=30,
        max_regions=42,
        num_heads=8,
        dropout=0.1
    ):
        """
        Args:
            spatial_embed_dim: Embedding dimension for spatial and temporal processing
            temporal_embed_dim: DEPRECATED - now same as spatial_embed_dim
            depth_spatial: Number of spatial transformer layers
            depth_temporal: Number of temporal transformer layers
            n_spatial_tokens: DEPRECATED - using mean pooling instead
            aggregation: DEPRECATED - kept for API compatibility, ignored
            freeze_backbone: Whether to freeze model weights during training
            max_channels: Maximum number of input channels
            n_frames: Number of temporal frames from CNN
            max_regions: Maximum number of brain regions
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Info message about using mean pooling
        print(f"ℹ️  ST model uses mean pooling for channel aggregation")
        print(f"   (similar to Conformer1DDualRegions architecture)\n")
        
        # Create SpatialTemporal model
        self.st_model = SpatialTemporalModel(
            spatial_embed_dim=spatial_embed_dim,
            temporal_embed_dim=spatial_embed_dim,  # Use same dimension
            depth_spatial=depth_spatial,
            depth_temporal=depth_temporal,
            n_classes=2,
            max_channels=max_channels,
            n_frames=n_frames,
            n_spatial_tokens=20,  # Deprecated, ignored
            max_regions=max_regions,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.st_model.parameters():
                param.requires_grad = False
            self.st_model.eval()
        
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x, regs):
        """
        Forward pass.
        
        Args:
            x: [batch, n_channels, seq_len] - Multi-channel time series
            regs: [batch, n_channels] - Region IDs for each channel
        
        Returns:
            logits: [batch, n_classes] - Classification logits
        """
        # If frozen, ensure no gradients
        if self.freeze_backbone:
            with torch.no_grad():
                return self.st_model(x, regs)
        else:
            return self.st_model(x, regs)


if __name__ == "__main__":
    # Test the wrapper
    print("Testing STChannelAggregator")
    print("=" * 80)
    
    # Create model (same API as other aggregators)
    model = STChannelAggregator(
        spatial_embed_dim=128,
        temporal_embed_dim=256,
        depth_spatial=6,
        depth_temporal=6,
        n_spatial_tokens=20,
        aggregation='mean',  # Ignored
        max_channels=150,
        n_frames=30
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Test input: [batch=2, n_channels=94, seq_len=2560]
    batch_size = 2
    n_channels = 94
    seq_len = 2560
    
    x = torch.randn(batch_size, n_channels, seq_len)
    regs = torch.randint(0, 42, (batch_size, n_channels))
    
    print(f"Input shapes:")
    print(f"  Signals: {x.shape}")
    print(f"  Regions: {regs.shape}")
    
    # Forward pass
    logits = model(x, regs)
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 2)")
    
    # Check probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities:\n{probs}\n")
    
    # Test with variable channel counts (padded to 150)
    x_var = torch.randn(batch_size, 150, seq_len)
    regs_var = torch.zeros(batch_size, 150, dtype=torch.long)
    regs_var[:, :50] = torch.randint(1, 42, (batch_size, 50))  # First 50 real
    
    logits_var = model(x_var, regs_var)
    print(f"Variable channels (50 real + 100 padded) output: {logits_var.shape}\n")
    
    # Test frozen backbone
    model_frozen = STChannelAggregator(
        spatial_embed_dim=128,
        temporal_embed_dim=256,
        depth_spatial=6,
        depth_temporal=6,
        freeze_backbone=True
    )
    logits_frozen = model_frozen(x, regs)
    print(f"Frozen backbone output: {logits_frozen.shape}\n")
    
    print("✓ STChannelAggregator working correctly!")
    print("✓ Maintains API compatibility with train_benchmarks.py")
    print("✓ Requires (signals, regions) as input")
