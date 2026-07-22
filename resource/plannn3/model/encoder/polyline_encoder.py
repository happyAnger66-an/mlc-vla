import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.GELU()])
            else:
                layers.extend(
                    [nn.Linear(c_in, mlp_channels[k], bias=False), LayerNorm(mlp_channels[k], bias=False), nn.GELU()]
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(c_in=in_channels, mlp_channels=[hidden_dim] * num_pre_layers, ret_before_act=False)
        self.mlps = build_mlps(
            c_in=hidden_dim * 2, mlp_channels=[hidden_dim] * (num_layers - num_pre_layers), ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points, C):
            polylines_mask (batch_size, num_polylines, num_points):

        Returns:
            feature (batch_size, num_polylines, hidden_dim or out_channels)
        """
        batch_size, num_polylines, num_points, C = polylines.shape

        # pre-mlp
        polylines_feature = self.pre_mlps(polylines)  # (N, na, nb, C)
        polylines_feature = torch.where(polylines_mask.unsqueeze(-1), polylines_feature, torch.zeros_like(polylines_feature))

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat(
            (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points, 1)), dim=-1
        )

        # mlp
        polylines_feature = self.mlps(polylines_feature)
        feature_buffers = torch.where(polylines_mask.unsqueeze(-1), polylines_feature, torch.zeros_like(polylines_feature))

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)

        # out-mlp
        if self.out_mlps is not None:
            valid_mask = polylines_mask.sum(dim=-1) > 0
            feature_buffers = self.out_mlps(feature_buffers)  # (N, C)
            feature_buffers = torch.where(valid_mask.unsqueeze(-1), feature_buffers, torch.zeros_like(feature_buffers))
        return feature_buffers
