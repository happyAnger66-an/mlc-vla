""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""
import math
from typing import List, Tuple, Optional, Union

import torch
from torch import nn as nn

def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.split(x.shape[-1]//2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rot_embed_cat(
        x: torch.Tensor,
        emb: torch.Tensor,
        half: bool = False
) -> torch.Tensor:
    sin_emb, cos_emb = emb.split(emb.shape[-1]//2, -1)
    # x: [..., D], eg [x0, x1, x2, x3, x4, x5]
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2
        # rope_rotate_half(x), eg [-x3, -x4, -x5, x0, x1, x2]
        return x * cos_emb + rope_rotate_half(x) * sin_emb
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x), eg [-x1, x0, -x3, x2, -x5, x4]
        return x * cos_emb + rot(x) * sin_emb

@torch.fx.wrap
def make_coords_dinov3(
        height: int,
        width: int,
        normalize_coords: str = 'separate',
        grid_indexing: str = 'ij',
        grid_offset: float = 0.,
        device: torch.device = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Make coordinate grid matching offset and normalization of original.
    Returns: coords with shape (HW, 2) in [-1, 1].
    """
    # 0.5-centered indices with optional offset
    coords_h = torch.arange(0.5, height, device=device, dtype=dtype) + grid_offset
    coords_w = torch.arange(0.5, width, device=device, dtype=dtype) + grid_offset

    # Normalization denominators
    if normalize_coords == "max":
        denom = float(max(height, width))
        h_denom = denom
        w_denom = denom
    elif normalize_coords == "min":
        denom = float(min(height, width))
        h_denom = denom
        w_denom = denom
    elif normalize_coords == "separate":
        h_denom = float(height)
        w_denom = float(width)
    else:
        raise ValueError(f"Unknown normalize_coords: {normalize_coords}")

    # Normalize to [0, 1]
    coords_h = coords_h / h_denom
    coords_w = coords_w / w_denom

    # Create grid then map to [-1, 1]
    if grid_indexing == "xy":
        grid_w, grid_h = torch.meshgrid(coords_w, coords_h, indexing="xy")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # (H, W, 2) -> (h, w order)
    else:
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # (H, W, 2)
    coords = coords.flatten(0, 1)  # (HW, 2)
    coords = 2.0 * coords - 1.0  # (H, W, 2) in [-1, 1]
    return coords


class RotaryEmbeddingDinoV3(nn.Module):
    """RoPE for timm DinoV3 port, numerically matching original.

    Math is aligned to original DinoV3 RopePositionEmbedding at https://github.com/facebookresearch/dinov3:
      - 0.5-centered coords normalized by H/W (or min/max), mapped to [-1,1]
      - training-time augmentations (shift/jitter/rescale)
      - periods schedule equals Rope's temperature (base) or min/max period
    """

    def __init__(
            self,
            dim: int,
            temperature: Optional[float] = 100.0,
            min_period: Optional[float] = None,
            max_period: Optional[float] = None,
            feat_shape: Optional[List[int]] = None,
            normalize_coords: str = "separate",  # 'min', 'max', 'separate'
            grid_offset: float = 0.0,
            grid_indexing: str = "ij",
            rotate_half: bool = True,
            shift_coords: Optional[float] = None,
            jitter_coords: Optional[float] = None,  # interpreted as factor J >= 1
            rescale_coords: Optional[float] = None,  # interpreted as factor R >= 1
    ):
        super().__init__()

        # Dimensions / output format
        self.dim = dim  # equal to head_dim for most vit applications
        self.rotate_half = rotate_half

        # Period schedule parameters
        self.temperature = float(temperature)
        self.min_period = min_period
        self.max_period = max_period

        # Coord processing + augs
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.aug_active = any([a is not None for a in [self.shift_coords, self.jitter_coords, self.rescale_coords]])

        # Grid config
        self.feat_shape = feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        # Precompute periods
        periods = self._compute_periods()
        self.register_buffer("periods", periods, persistent=False)

        if feat_shape is not None:
            self._cache_embed(feat_shape)
        else:
            self.register_buffer("pos_embed_cached", None, persistent=False)
            self.feat_shape = None

    def _compute_periods(self, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Construct periods from either min/max or temperature."""
        dim = self.dim // 4

        if self.min_period is not None and self.max_period is not None:
            exponents = torch.linspace(0, 1, dim, dtype=torch.float32)
            periods = self.min_period * ((self.max_period / self.min_period) ** exponents)
        else:
            if self.temperature is None:
                raise ValueError("Provide either min/max periods or `temperature`.")
            exponents = 2.0 * torch.arange(dim, device=device, dtype=dtype) / (self.dim // 2)
            periods = self.temperature ** exponents

        # NOTE: The original dinv3 model weights have periods downcast to bfloat16 in persistent buffers,
        # loaded models will differ a bit vs timm as periods is not persistent and generated in float32 by default

        return periods

    def _apply_coord_augs(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply shift/jitter/rescale train time augmentations."""
        if not self.training or not self.aug_active:
            return coords

        device = coords.device
        dtype = coords.dtype

        # Shift per-axis in [-s, +s]
        if self.shift_coords is not None:
            shift = float(self.shift_coords)
            shift_hw = torch.empty(2, device=device, dtype=dtype).uniform_(-shift, shift)
            coords = coords + shift_hw[None, :]

        # Jitter: per-axis log-uniform factor in [1/J, J]
        if self.jitter_coords is not None:
            jitter_factor = float(self.jitter_coords)
            if jitter_factor <= 0:
                raise ValueError("jitter_coords must be > 0 (interpreted as multiplicative factor).")
            jitter_max = math.log(jitter_factor)
            jitter_hw = torch.empty(2, device=device, dtype=dtype).uniform_(-jitter_max, jitter_max).exp()
            coords = coords * jitter_hw[None, :]

        # Rescale: shared scalar log-uniform factor in [1/R, R]
        if self.rescale_coords is not None:
            rescale_factor = float(self.rescale_coords)
            if rescale_factor <= 0:
                raise ValueError("rescale_coords must be > 0 (interpreted as multiplicative factor).")
            rescale_max = math.log(rescale_factor)
            rescale = torch.empty(1, device=device, dtype=dtype).uniform_(-rescale_max, rescale_max).exp()
            coords = coords * rescale

        return coords

    def _get_pos_embed_from_coords(self,  coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sin/cos embeddings with either 'half' or 'interleaved' layout."""
        # coords: (HW, 2); periods: (dim)
        dim = self.dim // 4
        device = self.periods.device
        dtype = self.periods.dtype
        assert self.periods.numel() == dim

        # NOTE this is a slightly later device/dtype switch than original
        coords = coords[:, :, None].to(device=device, dtype=dtype)
        angles = 2 * math.pi * coords / self.periods[None, None, :]
        angles = angles.flatten(1)  # (HW, dim // 2)

        if self.rotate_half:
            # Tile (half layout) (HW, dim // 2) -> (HW, dim)
            angles = angles.repeat(1, 2)
        else:
            # Interleaved layout (HW, dim // 2) -> (HW, dim)
            angles = angles.repeat_interleave(2, dim=-1)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin, cos

    def _create_embed(self, feat_shape: List[int], no_aug: bool = False) -> torch.Tensor:
        H, W = feat_shape
        coords = make_coords_dinov3(
            H, W,
            normalize_coords=self.normalize_coords,
            grid_indexing=self.grid_indexing,
            grid_offset=self.grid_offset
        )  # (HW, 2)
        if not no_aug:
            coords = self._apply_coord_augs(coords)
        sin, cos = self._get_pos_embed_from_coords(coords)  # 2 * (HW, dim)
        rope_embed = torch.cat([sin, cos], dim=-1)  # (HW, 2*dim)
        return rope_embed

    def _cache_embed(self, feat_shape: List[int]):
        rope_embed = self._create_embed(feat_shape, no_aug=True)  # create non-augmented embeds for cache
        self.register_buffer("pos_embed_cached", rope_embed, persistent=False)
        self.feat_shape = feat_shape

    def get_embed(self, shape: Optional[List[int]] = None) -> torch.Tensor:
        """Generate rope_embed matching DINOv3 RopePositionEmbedding numerics.

        Returns: (HW, num_heads, 2 * head_dim) with last dim = [sin, cos] cat.
        """
        if shape is not None:
            rope_embed = self._create_embed(shape)
        else:
            need_create = self.pos_embed_cached is None or (self.training and self.aug_active)
            if need_create:
                assert self.feat_shape is not None, 'feature shape must be cached on create'
                rope_embed = self._create_embed(self.feat_shape)
            else:
                assert self.pos_embed_cached is not None
                rope_embed = self.pos_embed_cached

        return rope_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get and apply rotary embeddings to x"""
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed, half=self.rotate_half)


def create_rope_embed(
        rope_type: str = 'cat',
        dim: int = 768,
        num_heads: int = 12,
        **kwargs
) -> nn.Module:
    """Factory function for creating rotary position embeddings.

    Args:
        rope_type: Type of RoPE to create. Options:
            - 'base': Basic RotaryEmbedding
            - 'cat': RotaryEmbeddingCat (concatenated sin/cos)
            - 'mixed': RotaryEmbeddingMixed (learnable per-depth frequencies)
            - 'dinov3': RotaryEmbeddingDinoV3 (with coordinate transforms)
        dim: Total embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments passed to the specific RoPE class

    Returns:
        Rotary embedding module
    """
    if rope_type == 'base':
        return RotaryEmbedding(dim=dim // num_heads, **kwargs)
    elif rope_type == 'cat':
        return RotaryEmbeddingCat(dim=dim // num_heads, **kwargs)
    elif rope_type == 'mixed':
        # Mixed requires depth parameter, generates differing embeddings per layer and head
        return RotaryEmbeddingMixed(dim=dim, num_heads=num_heads, **kwargs)
    elif rope_type == 'dinov3':
        return RotaryEmbeddingDinoV3(dim=dim // num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown RoPE type: {rope_type}")