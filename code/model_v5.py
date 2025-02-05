# %%
# imports
# autoreload
# %load_ext autoreload
# %autoreload 2
import torch
import torch.nn as nn
from torch import Tensor
from dataset import SzDatasetRegs
import math
from dataset import SzDatasetRegs
from einops import rearrange, repeat
from torch.utils.data import DataLoader

from model_config import MODEL_CONFIG
from lightly.models.modules import SwaVPrototypes, SwaVProjectionHead

PRINTER = False

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        # self.attend = nn.Softmax(dim=-1)
        self.attend = nn.LogSoftmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # Ensure mask is broadcastable to the shape of dots
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            mask = mask.expand(-1, self.heads, mask.shape[-1], -1)  # (batch, heads, seq_len, seq_len)
            # this makes the mask applied so that padded values are not considered in the softmax (only upper quadrant is ones)
            mask = torch.logical_and(
                mask, mask.transpose(-1, -2)
            ).to(torch.int)

            dots = dots.masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return self.norm(x)


# %%
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_spatial_len, max_temporal_len):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_spatial_len, max_temporal_len, dim))

    def forward(self, x, reg_idx):
        # return x + self.pos_embedding[reg_idx, temp_pos]
        return x + self.pos_embedding[reg_idx][:, :, :x.size(-2)]

# %%
class MultivarWav2Vec2(nn.Module):
    """
    Multivariate Wav2Vec2 model. This model takes in multiple 1D waveforms as input and combines the representations.
    The input waveforms are passed through individual 1D convolutional layers, and the representations are combined
    by taking the mean across the channels. The combined representations are then passed through a Wav2Vec2 transformer
    encoder.
    """

    def __init__(self, config, subsample=1):
        super(MultivarWav2Vec2, self).__init__()

        if subsample != 1:
            config.channel_buffer_size = int(config.channel_buffer_size * subsample)

        #  1D convolutions applied to each waveform, a series of 1D convolutions
        #  followed by layer normalization and GELU activation
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
            # transpose the output of the convolutional layer
            self.ft_enc.append(Transpose(1, 2))
            # layer normalization
            self.ft_enc.append(nn.LayerNorm(config.ft_enc_dims[i]))
            # GELU activation
            self.ft_enc.append(nn.GELU())
            # transpose the output of the convolutional layer
            self.ft_enc.append(Transpose(1, 2))

        # add a adaptive pool so different sized crops can be the same length afterward
        self.ft_enc.append(nn.AdaptiveAvgPool1d(config.frames))

        # convert the list of modules to a sequential module
        self.ft_enc = nn.Sequential(*self.ft_enc)

        self.fc_layer = nn.Linear(
            in_features=config.ft_enc_dims[-1] * config.channel_buffer_size,
            out_features=1,
        )

        # self.spatial_pos_encoder = PositionalEncoding3D(config.spatial_transformer_hidden, config.x_max, config.y_max, config.z_max)

        # # positional encoding
        # self.temporal_pos_encoder = PositionalEncoding(config.temporal_transformer_hidden, config.dropout)

        self.spatiotemporal_pos_encoder = PositionalEncoding(config.ft_enc_dims[-1], config.max_regs, config.frames)

        config.temporal_transformer_hidden = int(config.temporal_transformer_hidden * subsample)

        # add learnable class token for embeddings
        self.class_token = nn.Parameter(torch.randn(1, 1, config.temporal_transformer_hidden))

        self.projection_head = SwaVProjectionHead(config.temporal_transformer_hidden, 512, config.prototype_dim)

        self.prototypes = SwaVPrototypes(config.prototype_dim, n_prototypes=config.prototype_n,
                                         n_steps_frozen_prototypes=1)

        self.spatial_transformer = Transformer(config.spatial_transformer_hidden,
                                               config.spatial_transformer_blocks,
                                               config.spatial_transformer_heads,
                                               config.spatial_transformer_inner_heads,
                                               config.spatial_transformer_mlp_dim)

        self.temporal_transformer = Transformer(config.temporal_transformer_hidden,
                                                config.temporal_transformer_blocks,
                                                config.temporal_transformer_heads,
                                                config.temporal_transformer_inner_heads,
                                                config.temporal_transformer_mlp_dim)
        self.config = config
    def forward(self, x, regs):

        # x is a list of waveforms, each of shape (batch_size, num_channels, sequence_length)
        if PRINTER:
            print(f"x shape: {x.size()}")

        # computing eeg_channel wise convolution, reshape to put eeg_channels in batches so it can run faster
        batch_size = x.size(0)
        x = rearrange(x, 'b c n -> (b c) n').unsqueeze(1)

        if PRINTER:
            print(f"x shape after reshape: {x.size()}")
        x = self.ft_enc(x)
        if PRINTER:
            print(f"x shape after ft_enc: {x.size()}")

        # reshaped to put original batches back together
        x = rearrange(x, '(b c) d f -> b c d f', b=batch_size)
        if PRINTER:
            print(f"x shape after rearrange: {x.size()}")
        x = rearrange(x, 'b c d f -> b c f d')
        if PRINTER:
            print(f"x shape after rearrange: {x.size()}")

        # add spatiotemporal positional encoding
        x = self.spatiotemporal_pos_encoder(x, regs)
        if PRINTER:
            print(f"x shape after spatiotemporal pos encoder: {x.size()}")

        mask = (regs != 0)
        if PRINTER:
            print(f"mask shape: {mask.shape}")

        x = rearrange(x, 'b c f d -> b c (f d)')

        x = self.spatial_transformer(x, mask)
        if PRINTER:
            print(f"x shape after spatial transformer: {x.size()}")

        x = rearrange(x, 'b c (f d) -> b c f d', f=self.config.frames)

        if PRINTER:
            print(f"x shape after rearrange: {x.size()}")
        x = rearrange(x, 'b c f d -> b f c d')
        if PRINTER:
            print(f"x shape after rearrange: {x.size()}")
        x = rearrange(x, 'b c f d -> b f c d')
        x = rearrange(x, 'b c f d -> b f (c d)', f=self.config.frames)

        if PRINTER:
            print(f"x shape after rearrange: {x.size()}")

        # add class tokens
        cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=x.size(0))
        x = torch.cat((cls_tokens, x), dim=1)

        if PRINTER:
            print(f"x shape after cat: {x.size()}")

        x = self.temporal_transformer(x)
        if PRINTER:
            print(f"x shape after temporal transformer: {x.size()}")

        # recover class token
        x = x[:, 0]
        if PRINTER:
            print(f"x shape after recover class token: {x.size()}")

        # pass to swav projection head
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)

        if PRINTER:
            print(f"x shape after projection head: {x.size()}")

        return x

# %%
if __name__ == "__main__":
    PRINTER = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    if PRINTER:
        print(f"Using device: {device}")

    # use sz_dataset
    dataset = SzDatasetRegs()
    dataloader = DataLoader(
        dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=False, num_workers=4, persistent_workers=True
    )

    # Example usage
    config = MODEL_CONFIG  # Replace with your Wav2Vec2 configuration
    model = MultivarWav2Vec2(config)
    model = model.to(device)

    # get batch of 2
    batch = next(iter(dataloader))

    signals, regs, idx = batch

    regs = regs.long()

    high_resolution, low_resolution = signals[:2], signals[2:]

    high_resolution_feature = model(
        torch.nan_to_num(high_resolution[0].float(), posinf=0.0, neginf=0.0).to(device),
        regs.to(device))

# %%
