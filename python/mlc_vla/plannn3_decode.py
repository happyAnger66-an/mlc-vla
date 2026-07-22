"""plannn3 宿主侧输出解码：离散 traj token → waypoints（M3）。

TVM 图产出的是离散轨迹 token（``traj_ids``，`bit-exact` 对齐 golden 的主输出）；把它反解成
连续 waypoints 是**数据无关的纯 numpy 后处理**，按 arch.md 设计留在宿主侧（非 TVM engine）。

忠实复刻 ``resource/plannn3/model/encoder/trajectory_encoder_v2.py``：
- ``PCATokenizer.decode``：反量化 → PCA 重构 → 反 scale/归一（``[...,K] → [...,T,3]``）；
- ``TrajTokenizer.decode`` 的第二段：dx/dy/dyaw 增量 → 累积 yaw + line 重建 → 绝对 ``(x,y,yaw)``。

``planner_generate`` 里最终 ``action_ids = traj_ids[:, -main_action_length:]``（末 15 个 PCA 主 token），
前 ``meta_action_size`` 个是离散 meta-action（不进 PCA）。本模块据此拆分并反解。
"""

from __future__ import annotations

import json
import os
from typing import Optional  # noqa: UP035

import numpy as np


class PCATrajDecoder:
    """PCA 轨迹反解（复刻 ``PCATokenizer`` 的 decode 半程，纯 numpy）。

    从与训练一致的 ``pred_dxdydyaw_pca_tokenizer.json``（+ 同目录 basis npy）加载参数。
    """

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到 PCA tokenizer 配置: {config_path}")
        with open(config_path) as f:
            js = json.load(f)
        basis_filename = js.pop("basis_file", None)
        params = {}
        for k, v in js.items():
            if k == "original_shape":
                params[k] = list(v)
            elif isinstance(v, list):
                params[k] = np.array(v)
            else:
                params[k] = v
        if not basis_filename:
            raise ValueError("PCA 配置缺少 'basis_file'。")
        npy_path = os.path.join(os.path.dirname(config_path), basis_filename)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"PCA basis 未找到: {npy_path}")
        params["pca_components"] = np.load(npy_path)
        self.params = params
        self.n_bins = int(params["n_bins"])
        self.q_min = float(min(params["q_min"]))
        self.q_max = float(max(params["q_max"]))
        self.original_shape = list(params["original_shape"])  # [T, 3]

    def _dequantize(self, ids: np.ndarray) -> np.ndarray:
        ids = ids.astype(np.float32)
        codes_norm = ids / (self.n_bins - 1)
        return codes_norm * (self.q_max - self.q_min) + self.q_min

    def decode(self, token_ids: np.ndarray, num_components: Optional[int] = None) -> np.ndarray:
        """``token_ids [..., K]`` (int) -> ``[..., T, 3]`` 增量轨迹（dx,dy,dyaw）。"""
        token_ids = np.asarray(token_ids)
        target_k = num_components if num_components is not None else token_ids.shape[-1]
        token_ids = token_ids[..., :target_k]
        latent = self._dequantize(token_ids)
        comp = self.params["pca_components"][:target_k]
        x_recon_flat = np.dot(latent, comp) + self.params["pca_mean"]
        batch_dims = list(token_ids.shape[:-1])
        target_shape = batch_dims + self.original_shape
        x_recon_scaled = x_recon_flat.reshape(target_shape)
        x_recon_norm = x_recon_scaled / self.params["scales"]
        return x_recon_norm * (self.params["data_std"] + 1e-8) + self.params["data_mean"]


def reconstruct_waypoints(next_wp: np.ndarray) -> np.ndarray:
    """dx/dy/dyaw 增量 ``[B,N,3]`` -> 绝对 ``(x,y,yaw)`` ``[B,N,3]``。

    复刻 ``TrajTokenizer.decode`` 的第二段（numpy 版）：
        yaw       = cumsum(dyaw)
        delta_yaw = atan2(dy, dx)
        line_yaw  = [delta_yaw[0], delta_yaw[1:] + yaw[:-1]]
        len       = sqrt(dx^2+dy^2)
        x,y       = cumsum(len*cos(line_yaw)), cumsum(len*sin(line_yaw))
    """
    wp = np.asarray(next_wp, dtype=np.float64)
    dx, dy, dyaw = wp[:, :, 0], wp[:, :, 1], wp[:, :, 2]
    yaw = np.cumsum(dyaw, axis=1)
    delta_yaw = np.arctan2(dy, dx)
    line_yaw = np.concatenate([delta_yaw[:, 0:1], delta_yaw[:, 1:] + yaw[:, :-1]], axis=-1)
    line_length = np.sqrt(dx**2 + dy**2)
    x = np.cumsum(line_length * np.cos(line_yaw), axis=1)
    y = np.cumsum(line_length * np.sin(line_yaw), axis=1)
    return np.stack([x, y, yaw], axis=-1)


def split_traj_ids(traj_ids: np.ndarray, main_action_length: int = 15, meta_action_size: int = 3):
    """把 AR 产出的 ``traj_ids [B, pred_times]`` 拆成 (meta_action_ids, main_action_ids)。

    对齐 ``planner_generate``：main = 末 ``main_action_length`` 个；meta = main 之前的 ``meta_action_size`` 个。
    """
    ids = np.asarray(traj_ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    main = ids[:, -main_action_length:]
    meta = ids[:, -(main_action_length + meta_action_size):-main_action_length]
    return meta, main


def decode_traj_ids(
    traj_ids: np.ndarray,
    pca: Optional[PCATrajDecoder] = None,
    main_action_length: int = 15,
    meta_action_size: int = 3,
) -> dict:
    """完整输出解码：``traj_ids [B,pred_times]`` -> {meta_action, main_ids, [waypoints, wp_delta]}。

    - ``pca=None``：仅拆分 meta/main（无 PCA json 时的结构化路径）。
    - ``pca`` 给定：main token → PCA 反量化 → 增量轨迹 → 累积重建绝对 waypoints。
    """
    meta, main = split_traj_ids(traj_ids, main_action_length, meta_action_size)
    out = {"meta_action_ids": meta, "main_action_ids": main}
    if pca is not None:
        wp_delta = pca.decode(main, num_components=main_action_length)  # [B, T, 3]
        out["wp_delta"] = wp_delta
        out["waypoints"] = reconstruct_waypoints(wp_delta)  # [B, T, 3] (x,y,yaw)
    return out
