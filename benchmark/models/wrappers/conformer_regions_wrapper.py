"""
Wrapper for ConformerRegions model.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conformer.conformer_regions import ConformerRegions


class ConformerRegionsChannelAggregator(nn.Module):
    """
    Wrapper for ConformerRegions model.
    
    This model uses region embeddings instead of learned positional embeddings.
    Unlike the original Conformer, it requires region information as input.
    """
    
    def __init__(
        self,
        emb_size=128,
        depth=6,
        num_heads=8,
        aggregation='mean',  # Kept for API compatibility, not used
        freeze_backbone=False,
        max_channels=150,
        max_regions=42,
        dropout=0.1
    ):
        """
        Args:
            emb_size: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            aggregation: DEPRECATED - kept for API compatibility
            freeze_backbone: Whether to freeze model weights
            max_channels: Maximum number of input channels
            max_regions: Maximum number of brain regions
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create ConformerRegions model
        self.conformer_regions = ConformerRegions(
            emb_size=emb_size,
            depth=depth,
            num_heads=num_heads,
            n_classes=2,
            dropout=dropout,
            max_channels=max_channels,
            max_regions=max_regions
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.conformer_regions.parameters():
                param.requires_grad = False
            self.conformer_regions.eval()
        
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
        if self.freeze_backbone:
            with torch.no_grad():
                return self.conformer_regions(x, regs)
        else:
            return self.conformer_regions(x, regs)


if __name__ == "__main__":
    import torch
    
    # Test the wrapper
    print("Testing ConformerRegionsChannelAggregator")
    print("=" * 80)
    
    # Create model
    model = ConformerRegionsChannelAggregator(
        emb_size=128,
        depth=6,
        max_channels=150,
        max_regions=42
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Test input
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
    
    print("\n✓ ConformerRegionsChannelAggregator working correctly!")
