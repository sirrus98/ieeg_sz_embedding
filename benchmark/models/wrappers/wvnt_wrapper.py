"""
WVNT Channel Aggregator Wrapper.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from WVNT.model_adaptive import WaveModelAdaptive

# Handle both relative and absolute imports
try:
    from .base_wrapper import BaseChannelAggregator
except ImportError:
    from base_wrapper import BaseChannelAggregator


class WVNTChannelAggregator(BaseChannelAggregator):
    """
    Wrapper for WVNT model with masked channel aggregation.
    
    Architecture:
    - WaveModelAdaptive extracts features from each channel independently
    - Features are aggregated (mean or max) across channels with masking
    - Padded channels (regs == 0) are excluded from aggregation
    - MLP classifier produces final predictions
    """
    
    def __init__(self, checkpoint_path=None, aggregation='mean', freeze_backbone=True):
        """
        Args:
            checkpoint_path: Path to pretrained WVNT checkpoint (optional)
            aggregation: 'mean' or 'max' pooling across channels
            freeze_backbone: Whether to freeze WVNT weights during training
        """
        super().__init__(aggregation=aggregation)
        
        # Load WVNT model
        self.wvnt = WaveModelAdaptive()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Try to load from checkpoint
                # Use weights_only=False for compatibility with older checkpoints containing numpy objects
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.wvnt.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.wvnt.load_state_dict(checkpoint)
                print(f"✓ Loaded WVNT checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Failed to load checkpoint: {checkpoint_path}")
                print(f"  Error: {e}")
                print(f"  Training from scratch with random initialization")
        else:
            if checkpoint_path:
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print(f"  Training from scratch with random initialization")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.wvnt.parameters():
                param.requires_grad = False
            self.wvnt.eval()
        
        # MLP classifier on aggregated features
        # NEW: WVNT now outputs 256 features after fc1 (increased from 16)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def extract_features(self, x):
        """
        Extract features from WVNT model.
        
        Args:
            x: [batch*n_channels, 1, seq_len]
        
        Returns:
            features: [batch*n_channels, 256]
        """
        if self.wvnt.training:
            # If training (not frozen), compute gradients
            features = self.wvnt.get_features(x)
        else:
            # If frozen, no gradients
            with torch.no_grad():
                features = self.wvnt.get_features(x)
        
        return features
    
    def aggregate_channels_masked(self, features, batch_size, n_channels, mask):
        """
        Aggregate features across channels with masking to exclude padded channels.
        
        Args:
            features: [batch*n_channels, feat_dim]
            batch_size: original batch size
            n_channels: number of channels per sample
            mask: [batch, n_channels] - boolean mask where True = valid channel
        
        Returns:
            aggregated: [batch, feat_dim]
        """
        # Reshape to [batch, n_channels, feat_dim]
        features = features.view(batch_size, n_channels, -1)
        
        # Expand mask to match feature dimension: [batch, n_channels, 1]
        mask_expanded = mask.unsqueeze(-1).float()
        
        # Aggregate with masking
        if self.aggregation == 'mean':
            # Masked mean: sum only valid channels, divide by valid count
            features_masked = features * mask_expanded
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            aggregated = features_masked.sum(dim=1) / valid_counts
        elif self.aggregation == 'max':
            # Masked max: use torch.where to avoid clone (more efficient)
            # Set invalid channels to -inf without cloning
            features_masked = torch.where(
                mask.unsqueeze(-1),  # condition: valid channels
                features,            # true: use original features
                torch.tensor(-float('inf'), device=features.device, dtype=features.dtype)  # false: -inf
            )
            aggregated = features_masked.max(dim=1)[0]
        
        return aggregated
    
    def get_embeddings(self, x, regs=None):
        """
        Extract embeddings (features after aggregation, before classification).
        
        Args:
            x: [batch, n_channels, seq_len] - multi-channel time series
            regs: [batch, n_channels] - region IDs (0 = padded channel)
        
        Returns:
            embeddings: [batch, 256] - aggregated features
        """
        batch_size, n_channels, seq_len = x.shape
        
        # Create mask from regions: valid channels have regs != 0
        if regs is not None:
            mask = (regs != 0)  # [batch, n_channels]
        else:
            # Fallback: assume all channels are valid
            mask = torch.ones(batch_size, n_channels, dtype=torch.bool, device=x.device)
        
        # Reshape: channels → batch (use reshape for non-contiguous tensors)
        x = x.reshape(batch_size * n_channels, 1, seq_len)
        
        # Extract features
        features = self.extract_features(x)  # [batch*n_channels, feat_dim]
        
        # Aggregate channels with masking
        aggregated = self.aggregate_channels_masked(features, batch_size, n_channels, mask)
        
        return aggregated
    
    def forward(self, x, regs=None):
        """
        Forward pass with masked channel aggregation.
        
        Args:
            x: [batch, n_channels, seq_len] - multi-channel time series
            regs: [batch, n_channels] - region IDs (0 = padded channel)
        
        Returns:
            logits: [batch, n_classes] - classification logits
        """
        # Get embeddings
        aggregated = self.get_embeddings(x, regs)
        
        # Classify
        logits = self.classifier(aggregated)
        
        return logits


if __name__ == "__main__":
    # Test the wrapper with masked aggregation
    print("Testing WVNTChannelAggregator with Masked Aggregation")
    print("="*80)
    
    # Create model
    model = WVNTChannelAggregator(aggregation='mean', freeze_backbone=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Test 1: All valid channels
    print("Test 1: All valid channels (no padding)")
    batch_size = 2
    n_channels = 94
    seq_len = 2560
    
    x = torch.randn(batch_size, n_channels, seq_len)
    regs = torch.randint(1, 42, (batch_size, n_channels))  # All non-zero
    
    print(f"  Input shape: {x.shape}")
    print(f"  Regions shape: {regs.shape}")
    
    logits = model(x, regs)
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Logits: {logits}")
    print()
    
    # Test 2: With padded channels
    print("Test 2: With padded channels (94 real + 56 padded)")
    n_channels_padded = 150
    n_real = 94
    
    x_padded = torch.zeros(batch_size, n_channels_padded, seq_len)
    x_padded[:, :n_real, :] = torch.randn(batch_size, n_real, seq_len)
    
    regs_padded = torch.zeros(batch_size, n_channels_padded, dtype=torch.long)
    regs_padded[:, :n_real] = torch.randint(1, 42, (batch_size, n_real))
    
    print(f"  Input shape: {x_padded.shape}")
    print(f"  Regions shape: {regs_padded.shape}")
    print(f"  Valid channels per sample: {(regs_padded != 0).sum(dim=1).tolist()}")
    
    logits_padded = model(x_padded, regs_padded)
    print(f"  Output logits shape: {logits_padded.shape}")
    print(f"  Logits: {logits_padded}")
    print()
    
    # Test 3: Check for NaN/Inf
    has_nan = torch.isnan(logits_padded).any()
    has_inf = torch.isinf(logits_padded).any()
    
    if has_nan:
        print("⚠ WARNING: Outputs contain NaN!")
    elif has_inf:
        print("⚠ WARNING: Outputs contain Inf!")
    else:
        print("✓ Outputs are numerically stable (no NaN/Inf)")
    
    # Test 4: Test max aggregation
    print("\nTest 3: Max aggregation")
    model_max = WVNTChannelAggregator(aggregation='max', freeze_backbone=False)
    logits_max = model_max(x_padded, regs_padded)
    print(f"  Output logits shape: {logits_max.shape}")
    print(f"  Logits: {logits_max}")
    
    print("\n✓ WVNTChannelAggregator with masked aggregation working correctly!")
