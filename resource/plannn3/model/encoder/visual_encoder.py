import math
import numpy as np
import torch

from model.build import build_from_cfg


class VisualTokenizer(torch.nn.Module):
    def __init__(
        self,
        multiview_cams=None,
        temporal_multiview_configs=None,
        hidden_size=1024,
        sequence_size=4260,
        patch_size=16,
        enable_fold_fw=False,
        vqgan_cfg=None,
        video_shape=None,
        projection_point=False,
        add_pe=True,
        return_pe=False,
        point_encoder_cfg=None,
        mode="train",
        freeze=False,
        label_key=None,
    ):
        super().__init__()
        self.mode = mode
        self.bn_training = False
        self.sequence_size = sequence_size
        self.video_shape = video_shape
        self.patch_size = (patch_size, patch_size) if not isinstance(patch_size, (tuple, list)) else patch_size
        self.hidden_size = hidden_size
        self.label_key = label_key

        self.enable_fold_fw = enable_fold_fw
        self.vqgan = build_from_cfg(vqgan_cfg)

        self.multiview_cams = multiview_cams
        self.temporal_multiview_configs = temporal_multiview_configs

        self.view_tokens = torch.nn.Embedding(len(self.multiview_cams) * 2, hidden_size)

        self.projection_point = projection_point
        self.point_encoder = None
        self.add_pe = add_pe
        self.return_pe = return_pe
        if self.projection_point:
            self.point_encoder = build_from_cfg(point_encoder_cfg)
            self.pe_embedding = None
        
        for param in self.parameters():
            if param.requires_grad and freeze:
                param.requires_grad = False

    def build_pe_embedding(self, height, width, device, dtype):
        pe = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(height),
                    torch.arange(width),
                ]
            ), dim=-1
        ).to(dtype).to(device) + 0.5 # H, W, 2
        pe = pe.flatten(0, 1).unsqueeze(0) # 1, HW, 2
        # normalize
        pe[..., 0] = pe[..., 0] / height
        pe[..., 1] = pe[..., 1] / width
        
        pe_embedding = self.point_encoder(pe) # 1, HW, C
        return pe_embedding

    def forward(
        self,
        input_dict,
    ):
        image_data = input_dict["image"]
        batch_size = image_data[f"video:{self.multiview_cams[0]}"].shape[0]
        device = image_data[f"video:{self.multiview_cams[0]}"].device
        hist_img_feat = input_dict.get("hist_img_feat", None)

        hist_token_temporal = []
        if hist_img_feat is not None:
            prev = 0
            for temporal_idx, multiview_config in enumerate(self.temporal_multiview_configs[:-1]):
                multiview_cam_list, rescale_rate_list = multiview_config
                hist_token_multiview = {}
                for multiview_cam, rescale_rate in zip(multiview_cam_list, rescale_rate_list):
                    h, w = self.video_shape[multiview_cam][:2]
                    frame_token_len = int((h // self.patch_size[0] * rescale_rate[0]) * (w // self.patch_size[1] * rescale_rate[1]))
                    if prev+frame_token_len > hist_img_feat.shape[1]:
                        raise RuntimeError(f"the size of history image token is not matched !")
                    hist_token_multiview[multiview_cam] = hist_img_feat[:, prev:prev+frame_token_len]
                    prev += frame_token_len
                hist_token_temporal.append(hist_token_multiview)


        # batch encoder except fw
        fw_imgs = image_data['video:/camera/front/main']
        if self.enable_fold_fw:
            batch_cams = self.multiview_cams
            image_data['video:/camera/front/main'] = fw_imgs.reshape(*fw_imgs.shape[:-1], 3, -1).permute(0, 1, 4, 2, 3, 5).flatten(1, 2) # B, N*3, C, H, W//3
        else:
            batch_cams = [cam for cam in self.multiview_cams if cam != "/camera/front/main"]
        temporal_num_list = [image_data[f'video:{cam}'].shape[1] for cam in batch_cams]
        batch_img = torch.cat([image_data[f'video:{cam}'] for cam in batch_cams], dim=1) # B, L, C, H, W
        batch_img = batch_img.flatten(0, 1)
        loop_num = int(math.ceil(batch_img.shape[0] / 10))
        batch_token_list = []
        for loop_id in range(loop_num):
           start, end = loop_id * 10, (loop_id+1) * 10
           batch_token_list.append(
               self.vqgan(batch_img[start: end])
           )
        batch_token = torch.cat(batch_token_list, dim=0) # B*L, c, h, w 
        batch_token = batch_token.reshape(batch_size, -1, *batch_token.shape[1:])
        batch_token_split =  torch.split(batch_token, temporal_num_list, dim=1)
        token_dict = {cam: batch_token_split[ind] for ind, cam in enumerate(batch_cams)}
        if self.enable_fold_fw:
            fw_token = token_dict["/camera/front/main"] # B, N*3, c, h, w//3
            token_dict["/camera/front/main"] = fw_token.reshape(batch_size, -1, 3, *fw_token.shape[2:]).permute(0, 1, 3, 4, 2, 5).flatten(4, 5) # B, N, c, h, 3, w//3

        feat_list = []
        pe_list = []
        extra_output = {}
        for temporal_idx, multiview_config in enumerate(self.temporal_multiview_configs):
            multiview_cam_list, rescale_rate_list = multiview_config
            temporal_feat_list = []
            temporal_pe_list = []
            hist_token_multiview = {}
            for multiview_cam, rescale_rate in zip(multiview_cam_list, rescale_rate_list):
                cam_index = self.multiview_cams.index(multiview_cam)
                view_token = self.view_tokens(
                    torch.tensor([cam_index*2, cam_index*2+1], dtype=torch.int64, device=device)
                )[None].repeat(batch_size, 1, 1)
                temporal_feat_list.append(view_token[:, 0:1])
                temporal_pe_list.append(torch.zeros_like(temporal_feat_list[-1]))
                
                if hist_img_feat is not None and temporal_idx <= len(hist_token_temporal)-1:
                    token = hist_token_temporal[temporal_idx][multiview_cam]
                else:
                    if multiview_cam in token_dict:
                        positive_index = token_dict[multiview_cam].shape[1] - (len(self.temporal_multiview_configs) - temporal_idx)
                        token = token_dict[multiview_cam][:, positive_index]
                    else:
                        positive_index = image_data[f'video:{multiview_cam}'].shape[1] - (len(self.temporal_multiview_configs) - temporal_idx)
                        frame = image_data[f'video:{multiview_cam}'][:, positive_index]
                        if self.enable_fold_fw and multiview_cam == "/camera/front/main":
                            frame = frame.reshape(*frame.shape[:-1], 3, -1).permute(0, 3, 1, 2, 4).flatten(0, 1) # B*3, C, H, W//3
                        token = self.vqgan(frame)
                        if self.enable_fold_fw and multiview_cam == "/camera/front/main":
                            token = token.reshape(batch_size, 3, *token.shape[1:]).permute(0, 2, 3, 1, 4).flatten(3, 4) # B, c, h, w
                    if rescale_rate[0] != 1 or rescale_rate[1] != 1:
                        token = torch.nn.functional.interpolate(token, mode='bilinear', scale_factor=rescale_rate)
                    token_height, token_width = token.shape[2:]
                    token = token.flatten(2, 3).permute(0, 2, 1) # B, HW, C
                    hist_token_multiview[multiview_cam] = token

                if self.projection_point and temporal_idx == len(self.temporal_multiview_configs)-1 and multiview_cam == "/camera/front/main":
                    if self.pe_embedding is None:
                        self.pe_embedding = self.build_pe_embedding(token_height, token_width, token.device, token.dtype)
                    if self.add_pe:
                        token += self.pe_embedding

                temporal_feat_list.append(token)
                if self.return_pe and self.projection_point and temporal_idx == len(self.temporal_multiview_configs)-1 and multiview_cam == "/camera/front/main":
                    temporal_pe_list.append(self.pe_embedding.expand(token.shape[0], -1, -1).to(temporal_feat_list[-1]))
                else:
                    temporal_pe_list.append(torch.zeros_like(temporal_feat_list[-1]))
                temporal_feat_list.append(view_token[:, 1:2])
                temporal_pe_list.append(torch.zeros_like(temporal_feat_list[-1]))

            if len(hist_token_temporal) < temporal_idx + 1 and hist_token_multiview:
                hist_token_temporal.append(hist_token_multiview)

            temporal_feat_list = torch.cat(temporal_feat_list, dim=1)
            feat_list.append(temporal_feat_list)
            temporal_pe_list = torch.cat(temporal_pe_list, dim=1)
            pe_list.append(temporal_pe_list)

        embeds = torch.cat(feat_list, dim=1)
        pe = torch.cat(pe_list, dim=1)
        extra_output["pe"] = pe
        labels = None
        if self.label_key is not None:
            labels = input_dict.get(self.label_key, None)
        if embeds.shape[1] != self.sequence_size:
            raise RuntimeError(f"embeds sequence size {embeds.shape[1]} is not equal to {self.sequence_size}")

        # history for cache
        next_hist_img_feat = []
        if len(hist_token_temporal) != len(self.temporal_multiview_configs):
            raise RuntimeError(f"the shape of hist_token_temporal is wrong !")
        for temporal_idx, multiview_config in enumerate(self.temporal_multiview_configs[:-1]):
            multiview_cam_list, rescale_rate_list = multiview_config
            for multiview_cam, rescale_rate in zip(multiview_cam_list, rescale_rate_list):
                h, w = self.video_shape[multiview_cam][:2]
                frame_token_len = int((h // self.patch_size[0] * rescale_rate[0]) * (w // self.patch_size[1] * rescale_rate[1]))
                token = hist_token_temporal[temporal_idx+1][multiview_cam]
                if token.shape[1] != frame_token_len:
                    token = token.permute(0, 2, 1).reshape(-1, self.hidden_size, int(h // self.patch_size[0]), int(w // self.patch_size[1]))
                    token = torch.nn.functional.interpolate(token, mode='bilinear', scale_factor=rescale_rate)
                    token = token.flatten(2, 3).permute(0, 2, 1) # B, HW, C
                next_hist_img_feat.append(token)
        next_hist_img_feat = torch.cat(next_hist_img_feat, dim=1)
        if hist_img_feat is not None and hist_img_feat.shape[1] != next_hist_img_feat.shape[1]:
            raise RuntimeError(f"next_hist_img_feat shape is wrong, expect {hist_img_feat.shape}, but got {next_hist_img_feat.shape}")
        extra_output["hist_img_feat"] = next_hist_img_feat
        image_data['video:/camera/front/main'] = fw_imgs
        return embeds, labels, None, extra_output


    def bn_train(self, mode=True):
        self.bn_training = mode
        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train(mode=mode)
