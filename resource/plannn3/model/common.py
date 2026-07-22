import torch
from dataclasses import dataclass
from transformers import PretrainedConfig

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Generate sine position embedding from a position tensor.
    Args:
        pos_tensor (torch.Tensor): shape: [batch_size, N, 4]. the last dimension is [cx, cy, w, h] in
            normalized coordinates in range [0, 1].
        out_dim (int): the output dimension of the position embedding.
    Returns:
        pos (torch.Tensor): shape: [batch_size, N, out_dim].
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * 3.14159
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device="cuda")
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="trunc")) / half_hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")

    return pos

class NetConfig(PretrainedConfig):
    # net config
    def __init__(
        self,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        use_rmsnorm: bool = False,
        max_sequence_size: int = 2048,

        freeze_transformer: bool = False,
        module_list: list = None,

        speed_token_history_mask_prob: float = 0.0,
        speed_token_traj_offset: int = 3,

        # camera config
        resize_target_size: dict = None,
        crop_target_size: dict = None,
        multiview_cams: list = None,
        multiview_cams_index: list = None,
        fixed_proj_point: list = None, # (y, x)

        # kv cache
        use_kv_cache: bool = False,
        kv_cache_batch_size: int = 1,
        kv_cache_dtype: str = "float32",

        # forward func name for trace
        forward_func_name: str = "train_forward",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        self.use_rmsnorm = use_rmsnorm
        self.max_sequence_size = max_sequence_size

        self.freeze_transformer = freeze_transformer
        self.module_list = module_list

        self.speed_token_history_mask_prob = speed_token_history_mask_prob
        self.speed_token_traj_offset = speed_token_traj_offset

        # camera config
        self.resize_target_size = resize_target_size
        self.crop_target_size = crop_target_size
        self.multiview_cams = multiview_cams
        self.multiview_cams_index = multiview_cams_index
        self.fixed_proj_point = fixed_proj_point

        # kv cache
        self.use_kv_cache = use_kv_cache
        self.kv_cache_batch_size = kv_cache_batch_size
        self.kv_cache_dtype = kv_cache_dtype

        # forward func name for trace
        self.forward_func_name = forward_func_name
