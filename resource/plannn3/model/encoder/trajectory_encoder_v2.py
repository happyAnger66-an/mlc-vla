from __future__ import annotations

import os
import json
import torch
import logging
import numpy as np
import torch.nn.functional as F

from typing import Optional, ClassVar, List

logger = logging.getLogger(__name__)


class PCATokenizer:
    def __init__(self, config_path=None):
        """
        初始化 Tokenizer。
        如果提供了 config_path (json文件路径)，则自动加载模型。
        """
        self.params = {}
        self.is_fitted = False
        self.vocab_size = None
        
        if config_path:
            self.load_model(config_path)

    def __call__(self, trajectory, num_components=None):
        """Encode: [..., 25, 3] -> [..., n_components] (Token IDs)"""
        if not self.is_fitted: raise RuntimeError("Tokenizer 未训练")
        
        x = np.array(trajectory)
        input_shape = x.shape
        # if input_shape[-2:] != (25, 3) or input_shape[-2:] != (14, 3):
            # raise ValueError(f"输入维度错误，期望 [..., 25, 3]，实际 {input_shape}")
        
        target_k = num_components if num_components is not None else self.params['n_components']
            
        # 1. 预处理
        x_norm = (x - self.params['data_mean']) / (self.params['data_std'] + 1e-8)
        x_scaled = x_norm * self.params['scales']
        x_flat = x_scaled.reshape(*input_shape[:-2], -1)
        
        # 2. PCA 投影 (纯 Numpy 实现)
        # (X - mu) @ V.T
        x_centered = x_flat - self.params['pca_mean']
        latent = np.dot(x_centered, self.params['pca_components'][:target_k].T)
        
        # 3. 量化
        return self._quantize(latent, num_components=target_k)

    def decode(self, token_ids, num_components=None):
        """Decode: [..., n_components] (Token IDs) -> [..., 25, 3]"""
        if not self.is_fitted: raise RuntimeError("Tokenizer 未训练")
        
        token_ids = np.array(token_ids)
        target_k = num_components if num_components is not None else token_ids.shape[-1]
        
        # 截断输入以匹配 target_k
        token_ids = token_ids[..., :target_k]
        
        # 1. 反量化
        latent = self._dequantize(token_ids, num_components=target_k)
        
        # 2. PCA 重构
        # Z @ V + mu
        x_recon_flat = np.dot(latent, self.params['pca_components'][:target_k]) + self.params['pca_mean']
        
        # 3. 后处理
        current_batch_dims = list(token_ids.shape[:-1])
        ori_shape_dims = list(self.params['original_shape']) # [25, 3]
        target_shape = current_batch_dims + ori_shape_dims
        
        x_recon_scaled = x_recon_flat.reshape(target_shape)
        
        x_recon_norm = x_recon_scaled / self.params['scales']
        x_recon = x_recon_norm * (self.params['data_std'] + 1e-8) + self.params['data_mean']
        
        return x_recon

    def _quantize(self, latent_codes, num_components):
        q_min = self.q_min
        q_max = self.q_max
        n_bins = self.params['n_bins']
        
        codes_clipped = np.clip(latent_codes, q_min, q_max)
        codes_norm = (codes_clipped - q_min) / (q_max - q_min)
        ids = np.round(codes_norm * (n_bins - 1)).astype(np.int32)
        return np.clip(ids, 0, n_bins - 1)

    def _dequantize(self, ids, num_components):
        q_min = self.q_min
        q_max = self.q_max
        n_bins = self.params['n_bins']
        
        ids = ids.astype(np.float32)
        codes_norm = ids / (n_bins - 1)
        return codes_norm * (q_max - q_min) + q_min

    def load_model(self, json_path):
        """
        加载模型
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到配置文件: {json_path}")
            
        # 1. 加载 JSON
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        
        self.params = {}
        basis_filename = json_dict.pop('basis_file', None)
        
        # 2. 还原参数
        for k, v in json_dict.items():
            if k == 'original_shape':
                self.params[k] = list(v)
            elif isinstance(v, list):
                self.params[k] = np.array(v)
            else:
                self.params[k] = v
        
        # 3. 加载 NPY 基矩阵
        if basis_filename:
            npy_path = os.path.join(os.path.dirname(json_path), basis_filename)
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"配置文件引用了 {basis_filename}，但在 {npy_path} 未找到")
            
            self.params['pca_components'] = np.load(npy_path)
        else:
            raise ValueError("JSON 文件中缺少 'basis_file' 字段，无法加载 PCA 基")
            
        self.is_fitted = True
        self.vocab_size = self.params['n_bins']
        self.time_horizon = self.params['original_shape'][0]
        self.action_dim = self.params['original_shape'][1]
        self.q_min = min(self.params['q_min'])
        self.q_max = max(self.params['q_max'])
        
        print(f"Tokenizer 加载成功 | Config: {json_path} | Basis: {basis_filename}")


class TrajTokenizer(torch.nn.Module):
    def __init__(
        self,
        token_file="/data-algorithm-hl/lillian.xia/work/plannn2/tokenizer/my_fast_tokenizer_5s_1596_scale20_noq_addspecialtoken",
        vocab_size=2085,
        hidden_size=1024,
        traj_size=2048,
        begin_size=1,
        learnable_size=35,
        end_size=1,
        meta_action_size=2,
        sequence_size=36,
        use_default_prob=1.0,
        encode_type="dct",
        cat_hist_dxdydaw=False,
        hist_dxdydaw_num=4,
        mtp_last_num=None,
        mode="train",
    ) -> None:
        """
        embeds: waypoint embedding
        labels: next waypoint token ids
        masks: traj_masks
        """
        super().__init__()
        self.mode = mode
        self.sequence_size = 1 + learnable_size + meta_action_size
        self.use_default_prob = use_default_prob
        # self.mtp_last_num = mtp_last_num
        # if self.mtp_last_num is None:
        #     self.mtp_last_num = sequence_size

        self.has_begin_token = True
        self.traj_size = traj_size
        self.learnable_size = learnable_size
        self.pad_token_id = vocab_size - 1
        # self.begin_index = traj_size
        self.begin_main_token_id = self.traj_size
        self.begin_meta_action_token_id = self.traj_size + 1
        self.meta_action_size = meta_action_size
        self.vocab_size = vocab_size

        # if traj_size + begin_size + learnable_size + end_size != vocab_size:
        #     raise ValueError(f"{traj_size + begin_size + learnable_size + end_size} != {vocab_size} !")

        self.cat_hist_dxdydaw = cat_hist_dxdydaw
        self.hist_dxdydaw_num = hist_dxdydaw_num
        self.encode_type = encode_type
        if encode_type == "pca":
            self.vehicle_tokenizer = PCATokenizer(token_file)
        else:
            raise ValueError(
                f"This minimal inference example supports encode_type='pca' only, "
                f"got {encode_type!r}"
            )

        if hidden_size is not None:
            self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=self.pad_token_id)

        self.main_action_length = 15
        self.set_default = None # control default outside
            
    def forward(self, input_dict):
        batch_size = input_dict["batch_size"]
        traj_next_waypoint = input_dict.get("traj_next_waypoint", None)
        traj_masks = input_dict.get("egos_pred_mask", None)
        default_ids = torch.arange(
            self.traj_size,
            self.traj_size + self.sequence_size,
            device=self.embed_tokens.weight.device
        ).repeat(batch_size, 1)

        extra_output = {}
        if traj_next_waypoint is not None:
            if self.cat_hist_dxdydaw:
                his_traj_next_waypoint = input_dict.get("his_traj_next_waypoint", None) # B, N, 3
                assert his_traj_next_waypoint.shape[1] == self.hist_dxdydaw_num, f"hist_dxdydaw_num: {his_traj_next_waypoint.shape[1]} != {self.hist_dxdydaw_num}"
                traj_next_waypoint = torch.cat([his_traj_next_waypoint, traj_next_waypoint], dim=1)
            assert traj_next_waypoint.shape[1] == self.vehicle_tokenizer.time_horizon, f"{traj_next_waypoint.shape[1]} != {self.vehicle_tokenizer.time_horizon}"

            _, next_wp_ids = self.encode(traj_next_waypoint) # B, N, 3 -> B, 40
            
            # labels = torch.cat([torch.ones_like(next_wp_ids[:, :1]) * self.traj_size, next_wp_ids], dim=1)
            
            labels = []
            masks = []
            meta_action_labels = input_dict["meta_action_label"] # B, meta_action_size
            if meta_action_labels.shape[1] != self.meta_action_size:
                raise RuntimeError(f"meta_action_label shape error: {meta_action_labels.shape[1]} != {self.meta_action_size}")
            
            for sample_index, (_main_label, _meta_action) in enumerate(zip(next_wp_ids, meta_action_labels)):
                # token ids: <begin_meta_action_token>, <meta_action_i>..., <begin_main_token>, <main_0>, <main_1>, ..., <main_n>
                label = torch.cat(
                    [
                        _main_label.new_tensor([self.begin_meta_action_token_id]),
                        _meta_action.long(),
                        _main_label,
                    ],
                    dim=0,
                )
                mask = label.new_ones(label.size(0), dtype=torch.float32)
                
                assert len(label) == self.sequence_size, f"label length {len(label)} != sequence_size {self.sequence_size} !"
                
                labels.append(label)
                masks.append(mask)
            
            labels = torch.stack(labels, dim=0) # B, max_len
            masks = torch.stack(masks, dim=0) # B, max_len
            
            # 训练时，labels 包含 <begin_token> 和 main action token；推理时，labels 只包含 main action token
            masks = masks[:, 1:].contiguous() # B, N -> B, N-1
            
            rec_egos_gt_traj = self.decode(encode_tokens=next_wp_ids)[0][..., :2]
            extra_output = {"rec_egos_gt_traj": rec_egos_gt_traj}

            traj_ids = torch.clone(labels)
            # if self.training:
            #     default_random_mask = (
            #         torch.rand(batch_size, device=traj_next_waypoint.device)
            #         < self.use_default_prob
            #     )
            #     if self.set_default is not None:
            #         default_random_mask[0] = self.set_default
            #         self.set_default = None
            #     num = traj_ids.shape[1]
            #     traj_ids[default_random_mask, -self.mtp_last_num:] = default_ids[default_random_mask, -self.mtp_last_num:]

            embeds = self.embed_tokens(traj_ids) # B, N+1, C        
            pe = torch.zeros_like(embeds) # B, N+1, C
        else:
            labels = None
            traj_ids = default_ids
            embeds = self.embed_tokens(traj_ids) # B, N+1, C
            masks = torch.ones_like(embeds[:, :-1, 0])
            pe = torch.zeros_like(embeds) # B, N+1, C

        extra_output["pe"] = pe

        # masks = torch.ones_like(embeds[:, :-1, 0])
        masks = masks * (traj_masks.sum(-1, keepdim=True) > 0)

        if embeds.shape[1] > self.sequence_size:
            raise RuntimeError(f"embeds sequence size is bigger than {self.sequence_size} !")
        return embeds, labels, masks, extra_output
    
    def ids_to_embed(self, encode_tokens, is_begin_index=False, input_dict=None, keeplast=True):
        # encode_tokens: B, K
        extra_output = {}
        embed = self.embed_tokens(encode_tokens) # B, K, C
        if keeplast:
            embed = embed[:, -1:]
        return embed, extra_output

    def encode(self, next_wp_trajectory):
        device = next_wp_trajectory.device
        batch_ids = self.vehicle_tokenizer(
            next_wp_trajectory.cpu().numpy(),
            num_components=self.main_action_length,
        )
        batch_ids = torch.tensor(batch_ids, device=device).long()
        return next_wp_trajectory, batch_ids

    def decode(self, encode_tokens=None, encode_next_wps=None):
        if encode_tokens is not None:
            device = encode_tokens.device
            next_wp_trajectory = self.vehicle_tokenizer.decode(
                encode_tokens.cpu().numpy(),
                num_components=self.main_action_length,
            )
            next_wp_trajectory = torch.from_numpy(next_wp_trajectory).float().to(device)
            if self.cat_hist_dxdydaw:
                next_wp_trajectory = next_wp_trajectory[:, self.hist_dxdydaw_num:]
        elif encode_next_wps is not None:
            next_wp_trajectory = encode_next_wps


        input_dtype = next_wp_trajectory.dtype
        next_wp_trajectory = next_wp_trajectory.float()

        # next_wp_trajectory: B, N, 3
        yaw = torch.cumsum(next_wp_trajectory[:, :, 2], dim=1) # B, N 
        delta_yaw = torch.arctan2(next_wp_trajectory[:, :, 1], next_wp_trajectory[:, :, 0]) # B, N
        line_yaw = torch.cat([delta_yaw[:, 0:1], delta_yaw[:, 1:] + yaw[:, :-1]], dim=-1) # B, N
        line_length = torch.sqrt((next_wp_trajectory[:, :, :2]**2).sum(-1)) # B, N
        x_delta = line_length * torch.cos(line_yaw) # B, N
        y_delta = line_length * torch.sin(line_yaw) # B, N
        x = torch.cumsum(x_delta, dim=1)
        y = torch.cumsum(y_delta, dim=1)
        trajectory = torch.stack([x, y, yaw], dim=-1) # B, N, 3

        trajectory = trajectory.to(input_dtype)
        next_wp_trajectory = next_wp_trajectory.to(input_dtype)
        return trajectory, next_wp_trajectory
