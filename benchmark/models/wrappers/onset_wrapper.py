"""
ONSET Channel Aggregator Wrapper.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ONSET.model import LightweightSeizureDetector

# Handle both relative and absolute imports
try:
    from .base_wrapper import BaseChannelAggregator
except ImportError:
    from base_wrapper import BaseChannelAggregator


class ONSETChannelAggregator(BaseChannelAggregator):
    """
    Wrapper for ONSET model with channel aggregation.
    
    Architecture:
    - ONSET extracts features from each channel independently
    - Features are aggregated (mean or max) across channels
    - MLP classifier produces final predictions
    """
    
    def __init__(self, checkpoint_path=None, aggregation='mean', freeze_backbone=True):
        """
        Args:
            checkpoint_path: Path to pretrained ONSET checkpoint (optional)
            aggregation: 'mean' or 'max' pooling across channels
            freeze_backbone: Whether to freeze ONSET weights during training
        """
        super().__init__(aggregation=aggregation)
        
        # Load ONSET model
        self.onset = LightweightSeizureDetector()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Try to load from checkpoint
                # Use weights_only=False for compatibility with older checkpoints containing numpy objects
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                self.onset.load_state_dict(checkpoint)
                print(f"✓ Loaded ONSET checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Failed to load checkpoint: {checkpoint_path}")
                print(f"  Error: {e}")
                print(f"  Training from scratch with random initialization")
        else:
            # Initialize new model
            if checkpoint_path:
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print(f"  Training from scratch with random initialization")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.onset.parameters():
                param.requires_grad = False
            self.onset.eval()
        
        # MLP classifier on aggregated features
        # ONSET outputs base_filters=32 features after pooling
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    
    def extract_features(self, x):
        """
        Extract features from ONSET model.
        
        Args:
            x: [batch*n_channels, 1, seq_len]
        
        Returns:
            features: [batch*n_channels, 32]
        """
        if self.onset.training:
            # If training (not frozen), compute gradients
            features = self.onset.get_features(x)
        else:
            # If frozen, no gradients
            with torch.no_grad():
                features = self.onset.get_features(x)
        
        return features


if __name__ == "__main__":
    # Test the wrapper
    print("Testing ONSETChannelAggregator")
    print("="*60)
    
    # Create model
    model = ONSETChannelAggregator(aggregation='mean', freeze_backbone=False)
    
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
    
    print("✓ ONSETChannelAggregator working correctly!")
