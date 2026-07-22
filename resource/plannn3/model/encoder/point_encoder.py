import torch
from torch import nn

from model.common import gen_sineembed_for_position


class PointEncoderV2(nn.Module):
    def __init__(self, in_channels, hidden_dim, scale=(1.0, 1.0), num_head=1):
        super().__init__()
        """
        scale: normalize to (0, 1) range.
        """
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.scale = scale if isinstance(scale, tuple) else (scale, scale)

    def forward(self, points, points_mask=None):
        """
        Args:
            points (batch_size, num_points, 2):
            points_mask (batch_size, num_points):

        Returns:
            feature (batch_size, num_points, hidden_dim)
        """
        batch_size, num_points, C = points.shape
        scaled_points = points.float().clone()

        scaled_points[..., 0] *= self.scale[0]
        scaled_points[..., 1] *= self.scale[1]

        point_feature_pe = gen_sineembed_for_position(scaled_points, self.hidden_dim).to(scaled_points) # (B, N, D)
        if points_mask is not None:
            points_mask = points_mask.float()
            point_feature_pe = point_feature_pe * points_mask[..., None]

        if self.num_head > 1:
            point_feature_pe = point_feature_pe.unsqueeze(2).expand(-1, -1, self.num_head, -1).flatten(2, 3) # B, T, hd -> B, T, C 
        return point_feature_pe