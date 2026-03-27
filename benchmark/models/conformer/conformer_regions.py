"""
Conformer model with region-based positional encoding.

Similar to Conformer1D but uses region embeddings instead of learned positional embeddings.
This is a simpler architecture than dual transformers, using only a single transformer
with region information for spatial awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelCNNEncoder(nn.Module):
    """1D CNN for per-channel feature extraction."""
    
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, seq_len]
        Returns:
            [batch, embed_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.projection = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq_len, self.emb_size)
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, emb_size, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class ResidualAdd(nn.Module):
    """Residual connection with layer normalization."""
    
    def __init__(self, emb_size):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        
    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(self, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_size, dropout=dropout)
        self.residual_attn = ResidualAdd(emb_size)
        self.residual_ff = ResidualAdd(emb_size)
        
    def forward(self, x):
        x = self.residual_attn(x, self.attention)
        x = self.residual_ff(x, self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    
    def __init__(self, depth, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, dropout)
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConformerRegions(nn.Module):
    """
    Conformer with region-based positional encoding.
    
    Architecture:
    - Per-channel 1D CNN encoding
    - Region embeddings (instead of learned positional embeddings)
    - Single transformer encoder
    - CLS token for classification
    """
    
    def __init__(
        self,
        emb_size=128,
        depth=6,
        num_heads=8,
        n_classes=2,
        dropout=0.1,
        max_channels=150,
        max_regions=42
    ):
        """
        Args:
            emb_size: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            n_classes: Number of output classes
            dropout: Dropout rate
            max_channels: Maximum number of input channels
            max_regions: Maximum number of brain regions
        """
        super().__init__()
        self.emb_size = emb_size
        self.max_channels = max_channels
        self.max_regions = max_regions
        
        # Channel encoder
        self.channel_encoder = ChannelCNNEncoder(emb_size)
        
        # Region embeddings (replaces positional embeddings)
        self.region_embedding = nn.Embedding(max_regions, emb_size)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        # Transformer encoder
        self.transformer = TransformerEncoder(depth, emb_size, num_heads, dropout)
        
        # Classification head
        self.norm = nn.LayerNorm(emb_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        hidden_dim = emb_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x, regs):
        """
        Forward pass.
        
        Args:
            x: [batch, channels, seq_len] - Multi-channel time series
            regs: [batch, channels] - Region IDs for each channel
            
        Returns:
            [batch, n_classes] - Classification logits
        """
        batch, channels, seq_len = x.shape
        
        # Encode each channel independently
        x_reshaped = x.view(batch * channels, 1, seq_len)
        channel_embeds = self.channel_encoder(x_reshaped)  # [batch*channels, emb_size]
        channel_embeds = channel_embeds.view(batch, channels, self.emb_size)
        # Output: [batch, channels, emb_size]
        
        # Add region embeddings (instead of positional embeddings)
        region_embeds = self.region_embedding(regs)  # [batch, channels, emb_size]
        x = channel_embeds + region_embeds
        # Output: [batch, channels, emb_size]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Output: [batch, channels+1, emb_size]
        
        # Transformer
        x = self.transformer(x)
        # Output: [batch, channels+1, emb_size]
        
        # Extract CLS token
        cls_output = x[:, 0]
        # Output: [batch, emb_size]
        
        # Classify
        cls_output = self.norm(cls_output)
        cls_output = self.dropout_layer(cls_output)
        logits = self.classifier(cls_output)
        # Output: [batch, n_classes]
        
        return logits


if __name__ == "__main__":
    # Test the model
    print("Testing ConformerRegions")
    print("=" * 80)
    
    # Create model
    model = ConformerRegions(
        emb_size=128,
        depth=6,
        num_heads=8,
        n_classes=2,
        max_channels=150,
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
    
    print("\n✓ ConformerRegions working correctly!")
