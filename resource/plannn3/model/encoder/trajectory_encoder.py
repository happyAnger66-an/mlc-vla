import json
import logging

import numpy as np
import torch

from model.misc import point2corner

logger = logging.getLogger(__name__)


class Trajectory(torch.nn.Module):
    def __init__(
        self,
        token_file="/perception-hl/jin.chen4/world_model/trajectory_tokenizer/stage2_Gen_cluster2048_rules_realfilter_hierarchical_fix.json",
        use_corner=False,
        encoder_type="opt",
    ) -> None:
        super().__init__()

        if token_file.endswith(".json"):
            init_embedding = torch.Tensor(json.load(open(token_file))).float().cuda() # N, 3
        elif token_file.endswith(".npy"):
            init_embedding = torch.from_numpy(np.load(token_file)).float().cuda() # N, 3
        self.register_buffer("token_embedding", init_embedding)

        self.encoder_type = encoder_type
        logger.info(f"Trajectory encoder type: {self.encoder_type}")
        self.use_corner = use_corner
        if self.use_corner:
            logger.info("Use corner distance for trajectory tokenizer")
            token_corner = point2corner(init_embedding.unsqueeze(0)).squeeze(0).permute(1, 0, 2) # 4, N, 2
            self.register_buffer("token_corner", token_corner)
        else:
            self.token_corner = None

    @staticmethod
    def transfer_points(base_vector, target_vector):
        # base_vector, target_vector: B, 3
        # base_vector, target_vector are in the same coordinate, return new_vector of target_vector based on base_vector coordinate
        delta_vector = target_vector - base_vector
        x = delta_vector[:, 0] * torch.cos(base_vector[:, 2]) + delta_vector[:, 1] * torch.sin(base_vector[:, 2])
        y = -delta_vector[:, 0] * torch.sin(base_vector[:, 2]) + delta_vector[:, 1] * torch.cos(base_vector[:, 2])
        yaw = delta_vector[:, 2]
        new_vector = torch.stack([x, y, yaw], dim=-1)
        return new_vector

    def opt_encode(self, next_wp_trajectory, reverse=False):
        # next_wp_trajectory: B, N, 3(x, y, yaw)
        input_dtype = next_wp_trajectory.dtype
        next_wp_trajectory = next_wp_trajectory.float()
        encode_next_wps_list = [None] * next_wp_trajectory.shape[1]
        encode_tokens_list = [None] * next_wp_trajectory.shape[1]

        gap = torch.zeros_like(next_wp_trajectory[:, 0]) # B, 3(target -> real)
        for i in range(next_wp_trajectory.shape[1]):
            if reverse:
                ind = next_wp_trajectory.shape[1] - i - 1
            else:
                ind = i
            next_wp = next_wp_trajectory[:, ind]
            target_next_wp = self.transfer_points(gap, next_wp) # B, 3
            if self.use_corner:
                dist = torch.cdist(
                    point2corner(target_next_wp.unsqueeze(1)).squeeze(1).permute(1, 0, 2), # 4, B, 2
                    self.token_corner, # 4, N, 2
                ).mean(0) # B, N
            else:
                dist = torch.cdist(target_next_wp[:, :2], self.token_embedding[:, :2]) # B, N
            encode_token = dist.argmin(dim=1) # B
            real_next_wp = self.token_embedding[encode_token] # B, 3
            encode_tokens_list[ind] = encode_token
            encode_next_wps_list[ind] = real_next_wp
            gap = self.transfer_points(target_next_wp, real_next_wp)

        encode_tokens = torch.stack(encode_tokens_list, dim=1) # B, N
        encode_next_wps = torch.stack(encode_next_wps_list, dim=1).to(input_dtype) # B, N, 3
        return encode_next_wps, encode_tokens

    def simple_encode(self, next_wp_trajectory):
        # next_wp_trajectory: B, N, 3(x, y, yaw)
        batch, point_num = next_wp_trajectory.shape[:2]

        if self.use_corner:
            dist = torch.cdist(
                point2corner(next_wp_trajectory).flatten(0, 1).permute(1, 0, 2), # 4, B*K, 2
                self.token_corner, # 4, N, 2
            ).mean(0) # B*K, N
        else:
            dist = torch.cdist(next_wp_trajectory.float().flatten(0, 1)[:, :2], self.token_embedding[:, :2]) # B*K, N
        encode_tokens = dist.argmin(dim=1) # B*K
        encode_tokens = encode_tokens.reshape(batch, point_num) # B, K

        return next_wp_trajectory, encode_tokens

    def encode(self, next_wp_trajectory, reverse=False):
        if self.encoder_type == "opt":
            return self.opt_encode(next_wp_trajectory, reverse=reverse)
        else:
            return self.simple_encode(next_wp_trajectory)

    def ids_to_wp(self, encode_tokens):
        next_wp_trajectory = self.token_embedding[encode_tokens]
        return next_wp_trajectory

    def decode(self, encode_tokens=None, encode_next_wps=None):
        if encode_tokens is not None:
            next_wp_trajectory = self.token_embedding[encode_tokens]
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

class HisTrajTokenizer(torch.nn.Module):
    def __init__(
        self,
        token_file="/perception-hl/jin.chen4/world_model/trajectory_tokenizer/stage2_Gen_cluster2048_rules_realfilter_hierarchical_fix.json",
        vocab_size=2048,
        hidden_size=1024,
        sequence_size=1,
        use_ego_num=None,
        encoder_type="opt",
        use_corner=True,
        mask_prob=0.,
        mode="train",
    ) -> None:
        """
        embeds: waypoint embedding
        labels: next waypoint token ids
        masks: traj_masks
        """
        super().__init__()
        self.mode = mode
        self.sequence_size = sequence_size
        self.use_ego_num = use_ego_num or sequence_size
        self.Traj = Trajectory(token_file=token_file, use_corner=use_corner, encoder_type=encoder_type)
        self.vocab_size = vocab_size
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.mask_prob = mask_prob

    def forward(self, input_dict):
        batch_size = input_dict["batch_size"]
        traj_masks = input_dict.get("egos_his_mask", None) # B, N
        his_traj_next_waypoint = input_dict.get("his_traj_next_waypoint", None) # B, N, 3
        # his_egos_gt_traj = input_dict.get("his_egos_gt_traj", None) # B, N, 2

        labels = input_dict.get("speed_gap", None)
        masks = torch.ones_like(labels) if labels is not None else None

        if traj_masks is None:
            traj_masks = torch.ones_like(his_traj_next_waypoint[:, :, 0])

        if self.training and self.mask_prob > 0. and torch.rand(1).item() < self.mask_prob:
            traj_masks = torch.zeros_like(his_traj_next_waypoint[:, :, 0])
            masks = torch.zeros_like(labels) if labels is not None else None

        if self.mask_prob == 1:
            traj_masks = torch.zeros_like(his_traj_next_waypoint[:, :, 0])
            masks = torch.zeros_like(labels) if labels is not None else None

        if his_traj_next_waypoint is None:
            raise RuntimeError("his_traj_next_waypoint is None !")
        if his_traj_next_waypoint.shape[1] < self.sequence_size:
            raise RuntimeError(f"his_traj_next_waypoint shape error: {his_traj_next_waypoint.shape[1]} < {self.sequence_size} !")

        _, next_wp_ids = self.encode(his_traj_next_waypoint[:, -self.use_ego_num:]) # B, N
        
        if self.sequence_size > self.use_ego_num:
            next_wp_ids = torch.cat([next_wp_ids.new_tensor([self.vocab_size-1] * (self.sequence_size - self.use_ego_num)).unsqueeze(0).expand(batch_size, -1), next_wp_ids], dim=1)

        embeds = self.embed_tokens(next_wp_ids) # B, N, C

        embeds = embeds * traj_masks[:, -self.sequence_size:, None]

        extra_output = {}
        extra_output["pe"] = torch.zeros_like(embeds)

        if embeds.shape[1] != self.sequence_size:
            raise RuntimeError(f"embeds sequence size is not equal to {self.sequence_size}")
        return embeds, labels, masks, extra_output
    
    def encode(self, next_wp_trajectory, reverse=True):
        return self.Traj.encode(next_wp_trajectory, reverse=reverse)
