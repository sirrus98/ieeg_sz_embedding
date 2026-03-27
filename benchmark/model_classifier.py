"""
Classifier model for supervised fine-tuning.

Wraps the MultivarWav2Vec2 encoder with a classification head for:
1. Semiology classification (multi-class)
2. Ictal vs Interictal detection (binary)
"""

import sys
import os

# Add code directory to path for model imports  
code_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code')
if os.path.exists(code_dir):
    sys.path.insert(0, code_dir)

import torch
import torch.nn as nn

# Import model_config (safe - no file I/O)
try:
    from model_config import MODEL_CONFIG
except ImportError:
    # Fallback config based on model_config.py
    # Note: hidden dims must be divisible by num_heads
    class MODEL_CONFIG:
        frames = 20
        channel_buffer_size = 150
        max_regs = 41
        ft_enc_dims = (64, 128, 64, 32)
        ft_enc_strides = (5, 2, 2, 2)
        ft_enc_kernel_widths = (10, 3, 3, 2)
        spatial_transformer_blocks = 2
        spatial_transformer_hidden = 640  # frames * ft_enc_dims[-1] = 20 * 32, adjusted to 640 (divisible by 8)
        spatial_transformer_heads = 8  # Changed from 7 to 8 (640 / 8 = 80)
        spatial_transformer_inner_heads = 64
        spatial_transformer_mlp_dim = 2048
        temporal_transformer_blocks = 4
        temporal_transformer_hidden = 4800  # channel_buffer_size * ft_enc_dims[-1] = 150 * 32 (divisible by 8)
        temporal_transformer_heads = 8
        temporal_transformer_inner_heads = 64
        temporal_transformer_mlp_dim = 2048
        dropout = 0.1
        activation = "gelu"

# Note: We don't import model_v5.MultivarWav2Vec2 because it triggers config.py
# which has hardcoded paths. Instead, we reimplement the encoder below.


# Helper classes from model_v5.py
class Transpose(nn.Module):
    """Transpose tensor dimensions."""
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for spatial (region) and temporal (frame) positions.
    """
    def __init__(self, d_model, max_regions, max_frames):
        super().__init__()
        self.d_model = d_model
        self.max_regions = max_regions
        self.max_frames = max_frames
        
        # Learnable region embeddings
        self.region_embedding = nn.Embedding(max_regions + 1, d_model, padding_idx=0)
        
        # Learnable frame embeddings
        self.frame_embedding = nn.Embedding(max_frames, d_model)
    
    def forward(self, x, regions):
        """
        Args:
            x: (batch, channels, frames, d_model)
            regions: (batch, channels) - region indices
        """
        batch_size, n_channels, n_frames, d_model = x.shape
        
        # Add region embeddings
        region_emb = self.region_embedding(regions)  # (batch, channels, d_model)
        region_emb = region_emb.unsqueeze(2).expand(-1, -1, n_frames, -1)
        x = x + region_emb
        
        # Add frame embeddings
        frame_indices = torch.arange(n_frames, device=x.device)
        frame_emb = self.frame_embedding(frame_indices)  # (frames, d_model)
        frame_emb = frame_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, n_channels, -1, -1)
        x = x + frame_emb
        
        return x


class Transformer(nn.Module):
    """
    Transformer encoder block.
    """
    def __init__(self, hidden_dim, num_blocks, num_heads, inner_heads, mlp_dim):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.hidden_dim = hidden_dim
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - boolean mask (True for valid positions)
        """
        # Convert mask to attention mask format (additive mask)
        if mask is not None:
            # PyTorch expects: False for valid positions, True for positions to ignore
            attn_mask = ~mask
        else:
            attn_mask = None
        
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        return x


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
                
                # Make a copy of config to avoid modifying the original
                config = type('obj', (object,), {k: v for k, v in vars(config).items() if not k.startswith('__')})()
                
                if subsample != 1:
                    config.channel_buffer_size = int(config.channel_buffer_size * subsample)
                    config.temporal_transformer_hidden = int(config.temporal_transformer_hidden * subsample)
                
                # Ensure hidden dimensions are divisible by num_heads
                # Adjust num_heads if needed (easier than changing model architecture)
                if config.spatial_transformer_hidden % config.spatial_transformer_heads != 0:
                    # Find a suitable number of heads that divides the hidden dim
                    for heads in [8, 10, 16, 5, 4, 2, 1]:
                        if config.spatial_transformer_hidden % heads == 0:
                            print(f"  Note: Adjusted spatial_transformer_heads from {config.spatial_transformer_heads} to {heads}")
                            config.spatial_transformer_heads = heads
                            break
                
                if config.temporal_transformer_hidden % config.temporal_transformer_heads != 0:
                    for heads in [10, 12, 16, 8, 6, 5, 4, 2, 1]:
                        if config.temporal_transformer_hidden % heads == 0:
                            print(f"  Note: Adjusted temporal_transformer_heads from {config.temporal_transformer_heads} to {heads}")
                            config.temporal_transformer_heads = heads
                            break
                
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
                self.spatiotemporal_pos_encoder = PositionalEncoding(
                    config.ft_enc_dims[-1], config.max_regs, config.frames
                )
                
                # Transformers
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
        # Use weights_only=False for compatibility with older checkpoints containing numpy objects
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
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
