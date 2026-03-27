"""
WaveModel Adaptive - Modified WVNT model with adaptive pooling for variable-length inputs.

This version adds adaptive pooling to handle 2560-length (10-second @ 256Hz) inputs
instead of the original 128-length (1-second @ 128Hz) inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveModelAdaptive(nn.Module):
    """
    Redesigned WaveModel with deeper architecture and denser features.
    
    Architecture improvements:
    - 6 conv layers (instead of 3) for gradual feature extraction
    - Smaller kernels (7, 5, 3) without dilation for better gradient flow
    - BatchNorm for training stability
    - Adaptive pooling to 64 (not 16) for denser feature representation
    - 256-dim feature vector (instead of 16) for richer representations
    """
    
    def __init__(self):
        super().__init__()
        
        # Conv Block 1: 1 → 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.3)

        # Conv Block 2: 32 → 64
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.3)

        # Conv Block 3: 64 → 128
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.3)

        # Conv Block 4: 128 → 256 (no pooling)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.3)

        # Conv Block 5: 256 → 256 (no pooling)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.3)

        # Conv Block 6: 256 → 256 (no pooling)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop6 = nn.Dropout(0.3)

        # Adaptive pooling to fixed size for variable-length inputs
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)
        
        # FC layers with much larger feature dimension
        self.flatten_dim = 256 * 64  # 16384
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.drop_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        """
        Forward pass through 6-layer CNN architecture.
        
        Args:
            x: [batch, 1, seq_len] - univariate time series
               seq_len can be 128, 256, 2560, or any length
        
        Returns:
            logits: [batch, 2] - binary classification logits
        """
        # Conv Block 1: [batch, 1, seq_len] → [batch, 32, seq_len//4]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Conv Block 2: [batch, 32, seq_len//4] → [batch, 64, seq_len//16]
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Conv Block 3: [batch, 64, seq_len//16] → [batch, 128, seq_len//32]
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Conv Block 4: [batch, 128, seq_len//32] → [batch, 256, seq_len//32]
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop4(x)

        # Conv Block 5: [batch, 256, seq_len//32] → [batch, 256, seq_len//32]
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop5(x)

        # Conv Block 6: [batch, 256, seq_len//32] → [batch, 256, seq_len//32]
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop6(x)
        
        # Adaptive pooling to fixed size: [batch, 256, 64]
        x = self.adaptive_pool(x)
        
        # Flatten and classify: [batch, 16384] → [batch, 256] → [batch, 2]
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract features before final classification layer.
        Used for channel aggregation in wrapper models.
        
        Args:
            x: [batch, 1, seq_len] - univariate time series
        
        Returns:
            features: [batch, 256] - features after fc1
        """
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Conv Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop4(x)

        # Conv Block 5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop5(x)

        # Conv Block 6
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop6(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten and first FC
        x = x.flatten(1)
        features = F.relu(self.fc1(x))
        
        return features
    
    def predict_proba(self, x):
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_proba_temp(self, x):
        """Get temperature-scaled probability predictions."""
        logits = self.forward(x)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)


if __name__ == "__main__":
    # Test redesigned model with different input lengths
    model = WaveModelAdaptive()
    
    print("Testing Redesigned WaveModelAdaptive:")
    print("="*80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Test with original 128-length input
    x1 = torch.randn(4, 1, 128)
    out1 = model(x1)
    feat1 = model.get_features(x1)
    print(f"Test 1 - Original length (128):")
    print(f"  Input shape: {x1.shape}")
    print(f"  Output shape: {out1.shape} (expected: [4, 2])")
    print(f"  Features shape: {feat1.shape} (expected: [4, 256])")
    print()
    
    # Test with 2560-length input (10 seconds @ 256Hz)
    x2 = torch.randn(4, 1, 2560)
    out2 = model(x2)
    feat2 = model.get_features(x2)
    print(f"Test 2 - Target length (2560):")
    print(f"  Input shape: {x2.shape}")
    print(f"  Output shape: {out2.shape} (expected: [4, 2])")
    print(f"  Features shape: {feat2.shape} (expected: [4, 256])")
    print()
    
    # Test with arbitrary length
    x3 = torch.randn(4, 1, 1000)
    out3 = model(x3)
    feat3 = model.get_features(x3)
    print(f"Test 3 - Arbitrary length (1000):")
    print(f"  Input shape: {x3.shape}")
    print(f"  Output shape: {out3.shape} (expected: [4, 2])")
    print(f"  Features shape: {feat3.shape} (expected: [4, 256])")
    print()
    
    # Check for NaN/Inf
    has_nan = torch.isnan(out2).any() or torch.isnan(feat2).any()
    has_inf = torch.isinf(out2).any() or torch.isinf(feat2).any()
    
    if has_nan:
        print("⚠ WARNING: Model outputs contain NaN values!")
    elif has_inf:
        print("⚠ WARNING: Model outputs contain Inf values!")
    else:
        print("✓ Model outputs are numerically stable (no NaN/Inf)")
    
    print("✓ Redesigned model works with variable-length inputs!")
