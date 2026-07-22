import math
import torch
import numpy as np
import hashlib

class temporal_queue_cache:
    def __init__(self, tensor_shape=(1, 512, 1024), num=1, input_fre=5, output_fre=1, dtype="float"):
        self.dtype = dtype
        self.tensor_shape = tensor_shape
        if dtype == "float":
            self.dtype = torch.float32
        elif dtype == "int64":
            self.dtype = torch.int64
        else:
            raise ValueError(f"{dtype} is not supported !")

        if abs(input_fre / output_fre - np.round(input_fre / output_fre)) > 1e-2:
            raise ValueError(f"input_fre and output_fre do not match !")
            
        self.length = num * int(np.round(input_fre / output_fre))
        self.queue = [torch.zeros(self.tensor_shape, dtype=self.dtype).cuda()] * self.length

        self.indexes = [i for i in range(0, self.length, int(np.round(input_fre / output_fre)))]

    def put(self, current_feature):
        if isinstance(current_feature, np.ndarray):
            current_feature = torch.from_numpy(current_feature)

        self.queue.append(current_feature.to(self.dtype).cuda())
        if len(self.queue) > self.length:
            self.queue.pop(0)

    def pop(self):
        output_features = [self.queue[i] for i in self.indexes]
        if len(output_features) == 1:
            return output_features[0]
        return output_features

    def clear(self):
        self.queue = [torch.zeros(self.tensor_shape, dtype=self.dtype).cuda()] * self.length

def point2corner(points, length=4.79, width=1.97, wheelbase=2.888):
    """
    points: B, N, 3
    return: B, N, 4, 2
    """
    corners = torch.tensor(
        [
            [-(0.5 * length - 0.5 * wheelbase), -(0.5 * width)],
            [(0.5 * length + 0.5 * wheelbase), -(0.5 * width)],
            [(0.5 * length + 0.5 * wheelbase), (0.5 * width)],
            [-(0.5 * length - 0.5 * wheelbase), (0.5 * width)],
        ]
    )

    batch_size, num_points, _ = points.shape
    corners = corners.to(points).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1)

    cos = torch.cos(points[:, :, 2]) # B, N
    sin = torch.sin(points[:, :, 2]) # B, N
    rotation = torch.stack(
        [
            torch.stack([cos, sin], dim=2), # B, N, 2(cos, sin)
            torch.stack([-sin, cos], dim=2) # B, N, 2(-sin, cos)
        ],
        dim=2
    )

    corners = corners @ rotation + points[:, :, None, :2]
    return corners


def deterministic_uniform_prob(i: int, n: int = None, seed: str = "default") -> float:
    """
    给定整数 i（可选范围 n），返回一个确定性、均匀分布的伪随机概率值 ∈ [0, 1)
    - 对同一个 i 始终返回相同的概率
    - 不使用任何随机性
    - 分布近似 uniform(0, 1)
    """
    s = f"{i}-{seed}"
    h = hashlib.sha256(s.encode()).hexdigest()
    h_int = int(h, 16)
    # 最大值（2^256 - 1）
    max_val = 2**256 - 1
    return h_int / max_val


def is_extreme_float(x, upper_bound=1e6, lower_bound=-1e6):
    return math.isinf(x) or math.isnan(x) or x > upper_bound or x < lower_bound


if __name__ == "__main__":
    points = torch.tensor([[[0., 0., 0.], [1., 1., 0.], [0., 0., 0.1]]])
    corners = point2corner(points)
    print(corners)