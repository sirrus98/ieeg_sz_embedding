class ModelConfig:
    """
    Configuration parameters for the iEEG SwAV embedding model.

    All class-level attributes define the defaults; passing keyword arguments
    to __init__ overrides them on the instance without touching the class.
    """

    # --- temporal encoder ---
    frames = 20
    channel_buffer_size = 150
    max_regs = 41

    ft_enc_dims = (64, 128, 64, 32)
    ft_enc_strides = (5, 2, 2, 2)
    ft_enc_kernel_widths = (10, 3, 3, 2)

    # --- spatial transformer ---
    spatial_transformer_blocks = 2
    spatial_transformer_heads = 7
    spatial_transformer_inner_heads = 64
    spatial_transformer_mlp_dim = 2048

    # --- temporal transformer ---
    temporal_transformer_blocks = 4
    temporal_transformer_heads = 8
    temporal_transformer_inner_heads = 64
    temporal_transformer_mlp_dim = 2048

    dropout = 0.1
    activation = "gelu"

    # --- training ---
    batch_size = 32
    learning_rate = 1e-2
    epochs = 500

    # --- coordinate normalisation ---
    x_max = 300
    y_max = 300
    z_max = 300

    # --- SwAV prototype head ---
    prototype_dim = 128
    prototype_n = 512

    # --- multi-crop windows (time-domain sample counts at 256 Hz) ---
    # 10 s → 2 560 samples; default keeps existing behaviour
    multi_crop_config = [2000, 2000, 1000, 1000, 1000, 1000]

    use_spatial_pos_encoder = False

    def __init__(self, **kwargs):
        # Derived dims must be re-evaluated after any overrides, so copy class
        # defaults first, then apply overrides.
        self.frames = self.__class__.frames
        self.channel_buffer_size = self.__class__.channel_buffer_size
        self.max_regs = self.__class__.max_regs
        self.ft_enc_dims = self.__class__.ft_enc_dims
        self.ft_enc_strides = self.__class__.ft_enc_strides
        self.ft_enc_kernel_widths = self.__class__.ft_enc_kernel_widths
        self.spatial_transformer_blocks = self.__class__.spatial_transformer_blocks
        self.spatial_transformer_heads = self.__class__.spatial_transformer_heads
        self.spatial_transformer_inner_heads = self.__class__.spatial_transformer_inner_heads
        self.spatial_transformer_mlp_dim = self.__class__.spatial_transformer_mlp_dim
        self.temporal_transformer_blocks = self.__class__.temporal_transformer_blocks
        self.temporal_transformer_heads = self.__class__.temporal_transformer_heads
        self.temporal_transformer_inner_heads = self.__class__.temporal_transformer_inner_heads
        self.temporal_transformer_mlp_dim = self.__class__.temporal_transformer_mlp_dim
        self.dropout = self.__class__.dropout
        self.activation = self.__class__.activation
        self.batch_size = self.__class__.batch_size
        self.learning_rate = self.__class__.learning_rate
        self.epochs = self.__class__.epochs
        self.x_max = self.__class__.x_max
        self.y_max = self.__class__.y_max
        self.z_max = self.__class__.z_max
        self.prototype_dim = self.__class__.prototype_dim
        self.prototype_n = self.__class__.prototype_n
        self.multi_crop_config = list(self.__class__.multi_crop_config)
        self.use_spatial_pos_encoder = self.__class__.use_spatial_pos_encoder

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Derived sizes (re-computed after overrides)
        self.spatial_transformer_hidden = self.frames * self.ft_enc_dims[-1]
        self.temporal_transformer_hidden = self.channel_buffer_size * self.ft_enc_dims[-1]


# ---------------------------------------------------------------------------
# Per-duration crop configurations (sample counts at 256 Hz)
#   10 s  →  2 560 samples
#   20 s  →  5 120 samples
#   45 s  → 11 520 samples
#   full  →  variable; crops sized for ~45 s segments
# ---------------------------------------------------------------------------
DURATION_CONFIGS: dict = {
    "10s": {
        "multi_crop_config": [2000, 2000, 1000, 1000, 1000, 1000],
    },
    "20s": {
        "multi_crop_config": [3000, 3000, 1500, 1500, 1500, 1500],
    },
    "20s_peri": {
        "multi_crop_config": [3000, 3000, 1500, 1500, 1500, 1500],
    },
    "45s": {
        "multi_crop_config": [6000, 6000, 2000, 2000, 2000, 2000],
    },
    "full": {
        "multi_crop_config": [6000, 6000, 2000, 2000, 2000, 2000],
    },
}


def get_model_config(duration: str = "10s") -> ModelConfig:
    """Return a ModelConfig instance configured for *duration*.

    Args:
        duration: One of ``"10s"``, ``"20s"``, ``"20s_peri"``, ``"45s"``,
            or ``"full"``.

    Returns:
        A fresh :class:`ModelConfig` with crop windows set appropriately.
    """
    if duration not in DURATION_CONFIGS:
        raise ValueError(
            f"Unknown duration '{duration}'. "
            f"Choose from: {list(DURATION_CONFIGS)}"
        )
    return ModelConfig(**DURATION_CONFIGS[duration])


# Backward-compatible singleton (10 s defaults)
MODEL_CONFIG = get_model_config("10s")
