class MODEL_CONFIG:
    """
    This class contains all the configuration parameters for the project.
    """
    frames = 20  # number of frames
    channel_buffer_size = 150  # number of channels to buffer
    max_regs = 41  # maximum number of regions

    # feature extractor
    # V3 PARAMS
    ft_enc_dims = (64, 128, 64, 32)
    ft_enc_strides = (5, 2, 2, 2)
    ft_enc_kernel_widths = (10, 3, 3, 2)


    spatial_transformer_blocks = 2  # number of transformer blocks
    # spatial_transformer_hidden = 35  # transformer hidden size
    spatial_transformer_hidden = frames * ft_enc_dims[-1]  # transformer hidden size
    spatial_transformer_heads = 7  # transformer heads
    spatial_transformer_inner_heads = 64  # transformer inner size
    spatial_transformer_mlp_dim = 2048

    # default wav2vec2 config
    # transformer (implement base transformer)
    temporal_transformer_blocks = 4  # number of transformer blocks
    # temporal_transformer_hidden = 768  # transformer hidden size
    temporal_transformer_hidden = channel_buffer_size * ft_enc_dims[-1]  # transformer hidden size
    # transformer_hidden    = 768 // 16   # transformer hidden size
    temporal_transformer_heads = 8  # transformer heads
    temporal_transformer_inner_heads = 64  # transformer inner size
    temporal_transformer_mlp_dim = 2048
    dropout = 0.1  # dropout
    activation = "gelu"  # activation function

    # ft_enc_dims = (512, 512, 512, 512, 512, 512, 128)
    # ft_enc_strides = (5, 2, 2, 2, 2, 2, 2)
    # ft_enc_kernel_widths = (10, 3, 3, 3, 3, 2, 2)    

    # channel_buffer_size = 256  # number of channels to buffer

    # training parameters
    batch_size = 32  # batch size # larger batch size helps
    learning_rate = 1e-2  # learning rate
    epochs = 500  # number of epochs


    x_max = 300
    y_max = 300
    z_max = 300

    prototype_dim = 128
    prototype_n = 512

    # multi_crop_config = [6000, 6000, 2000, 2000, 2000, 2000]
    multi_crop_config = [2000, 2000, 1000, 1000, 1000, 1000]

    use_spatial_pos_encoder = False


    data_path = "/mnt/leif/littlab/users/pattnaik/cis522-final/data/processed_data"