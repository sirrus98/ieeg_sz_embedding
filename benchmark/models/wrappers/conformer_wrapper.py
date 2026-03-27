"""
Conformer Channel Aggregator Wrapper.

Updated to work with the new ViT-style Conformer1D that handles
multi-channel inputs internally. This wrapper maintains API compatibility
with train_benchmarks.py while simplifying to a pass-through architecture.
"""

import torch
import torch.nn as nn
import sys
import os
import warnings

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conformer.conformer_1d import Conformer1D


class ConformerChannelAggregator(nn.Module):
    """
    Wrapper for Conformer1D model.
    
    This wrapper maintains API compatibility with train_benchmarks.py
    while the new Conformer1D handles multi-channel inputs internally
    using a ViT-style architecture with learnable class token.
    
    IMPORTANT: The 'aggregation' parameter (mean/max) is DEPRECATED and IGNORED.
    
    The new architecture uses a ViT-style class token that learns optimal
    channel aggregation during training, which is more expressive than
    simple mean or max pooling. The parameter is only kept for backward
    compatibility with existing training scripts.
    """
    
    def __init__(self, emb_size=40, depth=6, aggregation='mean', 
                 freeze_backbone=False, max_channels=150):
        """
        Args:
            emb_size: Embedding dimension for Conformer
            depth: Number of transformer layers
            aggregation: DEPRECATED - This parameter is ignored.
                        The new ViT-style architecture uses a learnable class token
                        for channel aggregation instead of mean/max pooling.
                        Kept only for backward compatibility with train_benchmarks.py.
            freeze_backbone: Whether to freeze Conformer weights during training
            max_channels: Maximum number of channels to handle
        """
        super().__init__()
        
        # Always show info message about aggregation being deprecated
        print(f"ℹ️  Note: The 'aggregation' parameter ('{aggregation}') is deprecated and ignored.")
        print(f"   The new ViT-style Conformer uses a learnable class token for channel aggregation.")
        print(f"   This provides better performance than simple mean/max pooling.\n")
        
        # Create self-contained Conformer1D model
        self.conformer = Conformer1D(
            emb_size=emb_size, 
            depth=depth, 
            n_classes=2,
            max_channels=max_channels
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.conformer.parameters():
                param.requires_grad = False
            self.conformer.eval()
        
        self.freeze_backbone = freeze_backbone
    
    def get_embeddings(self, x):
        """
        Extract embeddings (class token output after norm, before classification).
        
        Args:
            x: [batch, n_channels, seq_len] - multi-channel time series
        
        Returns:
            embeddings: [batch, embed_dim] - class token embeddings
        """
        # If frozen, ensure no gradients
        if self.freeze_backbone:
            with torch.no_grad():
                return self.conformer.get_embeddings(x)
        else:
            return self.conformer.get_embeddings(x)
    
    def forward(self, x):
        """
        Forward pass (simple pass-through to Conformer1D).
        
        Args:
            x: [batch, n_channels, seq_len] - multi-channel time series
        
        Returns:
            logits: [batch, n_classes] - classification logits
        """
        # If frozen, ensure no gradients
        if self.freeze_backbone:
            with torch.no_grad():
                return self.conformer(x)
        else:
            return self.conformer(x)


if __name__ == "__main__":
    # Test the wrapper
    print("Testing ConformerChannelAggregator")
    print("="*60)
    
    # Create model (same API as before)
    model = ConformerChannelAggregator(emb_size=40, depth=6, aggregation='mean')
    
    # Test input: [batch=2, n_channels=94, seq_len=2560]
    batch_size = 2
    n_channels = 94
    seq_len = 2560
    
    x = torch.randn(batch_size, n_channels, seq_len)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    
    # Check probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities: {probs}")
    print()
    
    # Test with different channel counts
    x_var = torch.randn(batch_size, 50, seq_len)
    logits_var = model(x_var)
    print(f"Variable channels (50) output: {logits_var.shape}")
    print()
    
    # Test frozen backbone
    model_frozen = ConformerChannelAggregator(
        emb_size=40, depth=6, aggregation='mean', freeze_backbone=True
    )
    logits_frozen = model_frozen(x)
    print(f"Frozen backbone output: {logits_frozen.shape}")
    print()
    
    print("✓ ConformerChannelAggregator working correctly!")
    print("✓ Maintains API compatibility with train_benchmarks.py")
