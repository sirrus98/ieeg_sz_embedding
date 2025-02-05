"""
This version of the model is inspired by a VIT model. The model is a transformer model that takes in a 3D input and flattens and concatenates each channel
"""

# %%
import torch
import torch.nn as nn
from dataset import SzDatasetRegs
from einops import rearrange, repeat
from torch.utils.data import DataLoader

from model_config import MODEL_CONFIG
from lightly.models.modules import SwaVPrototypes, SwaVProjectionHead

# %%
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
            # mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            # mask = mask.expand(-1, self.heads, mask.shape[-1], -1)  # (batch, heads, seq_len, seq_len)
            mask = rearrange(mask, 'b n -> b () () n')
            mask = repeat(mask, 'b 1 1 n -> b h s n', h=self.heads, s=mask.size(-1))


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
class VITEEG(nn.Module):
    """
    VIT model for EEG data. The model is a transformer model that takes in a 3D input (batch_size, num_channels, sequence_length).
    The model is inspired by the Vision Transformer model. Each channel is flattened after tokenization and learned embeddings are added to the input.
    
    """

    def __init__(self, config):
        super(VITEEG, self).__init__()

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
        self.ft_enc.append(nn.AdaptiveAvgPool1d(config.spatial_transformer_hidden))

        # convert the list of modules to a sequential module
        self.ft_enc = nn.Sequential(*self.ft_enc)

        self.fc_layer = nn.Linear(
            in_features=config.ft_enc_dims[-1] * config.channel_buffer_size,
            out_features=1,
        )

        # add learnable class token for embeddings
        self.class_token = nn.Parameter(torch.randn(1, 1, config.ft_enc_dims[-1]))

        self.projection_head = SwaVProjectionHead(config.temporal_transformer_hidden, 512, config.prototype_dim)

        self.prototypes = SwaVPrototypes(config.prototype_dim, n_prototypes=config.prototype_n,
                                         n_steps_frozen_prototypes=1)

        self.temporal_transformer = Transformer(config.temporal_transformer_hidden,
                                                config.temporal_transformer_blocks,
                                                config.temporal_transformer_heads,
                                                config.temporal_transformer_inner_heads,
                                                config.temporal_transformer_mlp_dim)
        self.config = config
        self.pe = PositionalEncoding(config.ft_enc_dims[-1], config.max_regs, config.max_frames)

    def forward(self, x, regs):
        # x is a list of waveforms, each of shape (batch_size, num_channels, sequence_length)
        # print(f"x shape: {x.size()}")
    
        # computing eeg_channel wise convolution, reshape to put eeg_channels in batches so it can run faster
        batch_size = x.size(0)

        x = rearrange(x, 'b c n -> (b c) n').unsqueeze(1)

        # print(f"x shape after reshape: {x.size()}")
        x = self.ft_enc(x)
        # print(f"x shape after ft_enc: {x.size()}")

        # # set padding to negative infinity
        # zero_chs = (torch.reshape(coords, (-1, coords.size()[-1])) == 0).all(axis=(-1))
        # x[zero_chs.squeeze()] = -float('inf')
        # x[zero_chs.squeeze()] = 0
        # mask = (torch.reshape(coords, (-1, coords.size()[-1])) == 0).all(axis=(-1))
        # print(f"mask shape: {mask.shape}")

        # reshaped to put original batches back together
        x = rearrange(x, '(b c) d f -> b c d f', b=batch_size)
        # print(f"x shape after rearrange: {x.size()}")

        # rearrange to put channels and frames next to each other and flattened
        x = rearrange(x, 'b c d f -> b c f d')
        # print(f"x shape after rearrange: {x.size()}")

        # add positional encoding
        x = self.pe(x, regs)

        # print(f"x shape after positional encoding: {x.size()}")

        # x = rearrange(x, 'b c f d -> b (c f) d')
        # print(f"x shape after rearrange: {x.size()}")

        
        # # x = x.reshape((-1, self.config.spatial_transformer_hidden)).reshape(
        # #     (-1, self.config.temporal_transformer_hidden, self.config.spatial_transformer_hidden))

        # print(f"x shape after reshape: {x.size()}")        
        # mask = mask.reshape((-1, self.config.spatial_transformer_hidden)).reshape(
        #     (-1, self.config.temporal_transformer_hidden, self.config.spatial_transformer_hidden))

        # print(f"mask after reshape: {mask.shape}")

        # add spatial positional encoding
        # print(f"coords shape before spatial pos encoder: {coords.size()}")
        # print(f"x shape before spatial pos encoder: {x.size()}")
        # if self.config.use_spatial_pos_encoder:
        #     x = self.spatial_pos_encoder(x, coords)

        # print(f"x shape after spatial pos encoder: {x.size()}")

        # mask = (coords != 0).all(axis=-1)
        # mask_expanded = mask.repeat_interleave(int(self.config.temporal_transformer_hidden / mask.size(-1)), dim=-1)

        # print(f"mask shape: {mask.shape}")
        # print(f"mask_expanded shape: {mask_expanded.shape}")
        mask = regs != 0
        mask = repeat(mask, 'b c -> b c f', f=x.size(-2))
        # print(f"mask shape: {mask.size()}")

        x = rearrange(x, 'b c f d -> b (c f) d')
        mask = rearrange(mask, 'b c f -> b (c f)')

        # print(f"x shape after rearrange: {x.size()}")
        # print(f"mask shape after rearrange: {mask.size()}")

        cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=x.size(0))
        x = torch.cat((cls_tokens, x), dim=1)

        mask = torch.cat((torch.ones((x.size(0), 1), device=x.device), mask), dim=1)

        # print(f"x with cls tokens shape: {x.size()}")
        # print(f"mask shape: {mask.size()}")

        x = self.temporal_transformer(x, mask)

        # print(f"x shape after temporal transformer: {x.size()}")

        # pass to spatial encoder
        # print(f"spatial transformer input shape: {x.size()}")

        # x = self.spatial_transformer(x, mask_expanded)

        # x = x.transpose(1, 2)

        # # add class tokens
        # cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=x.size(0))
        # x = torch.cat((cls_tokens, x), dim=1)
        # # add temporal positional encoding
        # x = self.temporal_pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        # # pass to temporal encoder
        # x = self.temporal_transformer(x)
        # # recover class token
        x = x[:, 0]

        # pass to swav projection head
        x = self.projection_head(x)

        x = nn.functional.normalize(x, dim=1, p=2)

        return x

# %%
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # print(f"Using device: {device}")

    # use sz_dataset
    dataset = SzDatasetRegs()
    dataloader = DataLoader(
        dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=False, num_workers=4, persistent_workers=True
    )

    # Example usage
    config = MODEL_CONFIG  # Replace with your Wav2Vec2 configuration
    model = VITEEG(config)
    model = model.to(device)

    # get batch of 2
    batch = next(iter(dataloader))

    signals, regs, idx = batch

    regs = regs.long()

    high_resolution, low_resolution = signals[:2], signals[2:]

    high_resolution_feature = model(torch.nan_to_num(high_resolution[0].float(), posinf=0.0, neginf=0.0).to(device)
                                    , regs)
    
    # high_resolution_features = [model(torch.nan_to_num(x.float(), posinf=0.0, neginf=0.0).to(device)
    #                                         , coords) for x in high_resolution]
    


# %%
