import math
import copy
import logging
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import inspect
from typing import Tuple, Optional
from transformers import PreTrainedModel

from model.common import NetConfig
from model.build import build_from_cfg

logger = logging.getLogger(__name__)


def apply_rotary_emb_interleaved(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        head_dim = config.n_embd // config.n_head
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head dim, got {head_dim}")
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        block_size = config.max_sequence_size
        self.head_dim = head_dim
        cos, sin = self._precompute_rotary_embeddings(block_size, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = torch.tril(torch.ones(block_size, block_size)).bool()    # 标准下三角矩阵
        bias = bias.view(1, 1, block_size, block_size)
        self.register_buffer("bias", bias)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin

    def _apply_rope(self, q, k, position_ids):
        # q, k: B x nh x T x hs ; position_ids: T
        position_ids = position_ids.to(device=q.device, dtype=torch.long)
        cos = self.cos[:, :, position_ids, :].to(dtype=q.dtype)
        sin = self.sin[:, :, position_ids, :].to(dtype=q.dtype)
        q = apply_rotary_emb_interleaved(q, cos, sin)
        k = apply_rotary_emb_interleaved(k, cos, sin)
        return q, k

    def forward(self, x, padding_index, input_pos=None, pe=None, attention_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) 
        position_ids = (
            input_pos
            if input_pos is not None
            else torch.arange(T, device=x.device, dtype=torch.long)
        )
        q, k = self._apply_rope(q, k, position_ids)

        bias = self.bias
        _, _, block_size, _ = bias.shape
        bias = bias.expand(B, 1, block_size, block_size)

        bias = bias[:, :, :T, :T]

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=bias.device, dtype=torch.bool)
            if input_pos is not None:
                attention_mask = attention_mask[:, :, input_pos, : input_pos[-1] + 1]
            else:
                attention_mask = attention_mask[:, :, :T, :T]
            bias = bias & attention_mask

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(bias.logical_not(), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, padding_index=None, input_pos=None, pe=None, attention_mask=None):
        x = x + self.attn(self.ln_1(x), padding_index, input_pos=input_pos, pe=pe, attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Net(PreTrainedModel):
    config_class = NetConfig
    def __init__(self, model_args):
        if isinstance(model_args, NetConfig):
            config = model_args
        else:
            config = NetConfig(**model_args)
        super(Net, self).__init__(config)

        self.config = config
        logger.info(f"max_sequence_size: {config.max_sequence_size}")


        self.module_names = []
        for module_cfg in config.module_list:
            name = module_cfg["name"]
            self.module_names.append(name)

            encoder_cfg = module_cfg["encoder"]
            head_cfg = module_cfg.get("head", None)
            loss_cfg = module_cfg.get("loss", None)
            setattr(self, f"{name}_encoder", build_from_cfg(encoder_cfg) if encoder_cfg is not None else None)
            setattr(self, f"{name}_head", build_from_cfg(head_cfg) if head_cfg is not None else None)
            setattr(self, f"{name}_loss", build_from_cfg(loss_cfg) if loss_cfg is not None else None)

        
        self.transfomer = nn.ModuleDict(
            dict(
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            )
        )

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.apply(self._init_weights)

        self.get_num_params()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        transformer_params = sum(p.numel() for p in self.transfomer.parameters())
        logger.info("number of transformer parameters: %.2fM" % (transformer_params / 1e6,))

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            for name, module in self.named_modules():
                if isinstance(module, nn.Embedding):
                    n_params -= module.weight.numel()
        
        logger.info("total number of parameters: %.2fM" % (n_params / 1e6,))
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def build_speed_token_history_attention_mask(self, embeds_sequence_size, batch_size, device):
        mask_prob = self.config.speed_token_history_mask_prob
        if not self.training or mask_prob <= 0.0:
            return None
        if "history_traj" not in self.module_names or "traj" not in self.module_names:
            return None
        sample_mask = torch.rand(batch_size, device=device) < mask_prob
        if not sample_mask.any():
            return None

        history_index = self.module_names.index("history_traj")
        traj_index = self.module_names.index("traj")
        history_start = sum(embeds_sequence_size[:history_index])
        history_end = history_start + embeds_sequence_size[history_index]
        traj_start = sum(embeds_sequence_size[:traj_index])
        speed_token_pos = traj_start + self.config.speed_token_traj_offset
        speed_predict_pos = max(traj_start, speed_token_pos - 1)
        sequence_size = sum(embeds_sequence_size)

        attention_mask = torch.ones(batch_size, 1, sequence_size, sequence_size, dtype=torch.bool, device=device)
        attention_mask[sample_mask, :, speed_predict_pos : speed_token_pos + 1, history_start:history_end] = False
        return attention_mask

    def train(self, mode=True):
        super().train(mode)

        if self.config.freeze_transformer:
            logger.info(f"freeze transformer")
            self.transfomer.eval()
            for param in self.transfomer.parameters():
                param.requires_grad = False

        for module_cfg in self.config.module_list:
            name = module_cfg["name"]
            encoder = getattr(self, f"{name}_encoder")
            freeze_encoder = module_cfg.get("freeze_encoder", False)
            freeze_encoder_bn = module_cfg.get("freeze_encoder_bn", False)
            if freeze_encoder:
                logger.info(f"freeze {name}_encoder")
                encoder.eval()
                for param in encoder.parameters():
                    param.requires_grad = False
            elif freeze_encoder_bn:
                logger.info(f"freeze {name}_encoder_bn")
                for module in encoder.modules():
                    if isinstance(
                        module,
                        (torch.nn.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm),
                    ):
                        module.eval()

            head = getattr(self, f"{name}_head")
            freeze_head = module_cfg.get("freeze_head", False)
            if freeze_head and head is not None:
                logger.info(f"freeze {name}_head")
                head.eval()
                for param in head.parameters():
                    param.requires_grad = False

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type="cuda"):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(self, *args, **kwargs):
        return getattr(self, f"{self.config.forward_func_name}")(*args, **kwargs)

    def train_forward(
        self,
        image: torch.LongTensor = None,
        subpath_polylines: torch.FloatTensor = None,
        subpath_polylines_mask: torch.IntTensor = None,
        navi_info: torch.IntTensor = None,
        navi_label: Optional[torch.LongTensor] = None,
        egos_gt_traj: Optional[torch.FloatTensor] = None,
        egos_pred_mask: Optional[torch.FloatTensor] = None,
        traj_next_waypoint: Optional[torch.FloatTensor] = None,
        his_egos_gt_traj: Optional[torch.FloatTensor] = None,
        egos_his_mask: Optional[torch.FloatTensor] = None,
        his_traj_next_waypoint: Optional[torch.FloatTensor] = None,
        hist_img_tokens: Optional[torch.LongTensor] = None,
        hist_img_feat: Optional[torch.LongTensor] = None,
        cam_extrinsics: Optional[torch.FloatTensor] = None,
        meta_action_label: Optional[torch.LongTensor] = None,
        cali_path=None,
        projection_interface=None,
        speed_gap: Optional[torch.FloatTensor] = None,
        traj_min_dist_label=-100,
    ):
        batch_size = image['video:/camera/front/main'].shape[0] 
        input_dict = {
            "image": image, # dict, value: B, N, 3, H, W
            "batch_size": batch_size,
            "traj_next_waypoint": traj_next_waypoint,
            "egos_gt_traj": egos_gt_traj,
            "egos_pred_mask": egos_pred_mask,
            "his_traj_next_waypoint": his_traj_next_waypoint,
            "his_egos_gt_traj": his_egos_gt_traj,
            "egos_his_mask": egos_his_mask,
            "hist_img_tokens": hist_img_tokens,
            "hist_img_feat": hist_img_feat,
            "subpath_polylines": subpath_polylines,
            "subpath_polylines_mask": subpath_polylines_mask,
            "navi_info": navi_info,
            "navi_label": navi_label,
            "cam_extrinsics": cam_extrinsics,
            "cali_path": cali_path,
            "projection_interface": projection_interface,
            "speed_gap": speed_gap,
            "meta_action_label": meta_action_label,
            "traj_min_dist_label": traj_min_dist_label,
        }

        output = {}
        embeds_list = []
        labels_list = []
        masks_list = []
        for name in self.module_names:
            embeds, labels, masks, extra_outputs = getattr(self, f"{name}_encoder")(input_dict)
            embeds_list.append(embeds)
            labels_list.append(labels)
            masks_list.append(masks)
            if extra_outputs is not None and "pe" in extra_outputs:
                extra_outputs.pop("pe")
            if extra_outputs is not None:
                output.update(extra_outputs)

        embeds_sequence_size = [embeds.shape[1] for embeds in embeds_list]
        inputs_embeds = torch.cat(embeds_list, dim=1)
        attention_mask = self.build_speed_token_history_attention_mask(
            embeds_sequence_size,
            batch_size,
            inputs_embeds.device,
        )

        x = inputs_embeds
        for layer in self.transfomer["h"]:
            x = layer(x, attention_mask=attention_mask)
        hidden_states = x
        
        hidden_states_list = torch.split(hidden_states, embeds_sequence_size, dim=1)
        for index, name in enumerate(self.module_names):
            hidden_state = hidden_states_list[index]
            label = labels_list[index]
            mask = masks_list[index]
            if getattr(self, f"{name}_head"):
                head_outputs = getattr(self, f"{name}_head")(hidden_state)
                output.update(
                    {f"{name}_{key}": value for key, value in head_outputs.items()}
                )

                if name == "traj":
                    pred_egos_gt_traj, predict_next_wps = self.traj_encoder.decode(encode_tokens=head_outputs["pred_main_action_ids"])
                    output["wp_outputs"] = pred_egos_gt_traj[..., :2] # B, N, 2

            if getattr(self, f"{name}_loss") and self.training:
                loss_dict = getattr(self, f"{name}_loss")(head_outputs, label, mask)
                output.update(
                    {f"{name}_{key}": value for key, value in loss_dict.items()}
                )
        return output

    def trace_encoder_forward(
        self,
        pinhole_raw_img,
        # svc_raw_img,
        hist_img_feat,
        subpath_polylines,
        subpath_polylines_mask,
        navi_info,
        hist_action,
    ):
        raw_projection_point = hist_action[0, 4, :2] # 2(y, x)
        hist_action = hist_action[:, :4] # only use the first 4 waypoints as input, since the 5th point is used for cropping cfg computation

        def resize_crop_image(img, resize_target_size, crop_target_size):
            resize_img = F.interpolate(img.unsqueeze(0), size=resize_target_size, mode="nearest").squeeze(0)
            if len(crop_target_size) == 2:
                height, width = crop_target_size
                top = (resize_target_size[0] - crop_target_size[0]) // 2
                left = (resize_target_size[1] - crop_target_size[1]) // 2
            elif len(crop_target_size) == 4:
                height, width, top, left = crop_target_size
            crop_img = resize_img[:, top:top+height, left:left+width]
            return crop_img

        # compute crop cfg
        # can not convert projection_point to python scalar
        raw_h, raw_w = pinhole_raw_img.shape[2:]

        fw_cam = "/camera/front/main"
        resize_scale = (
            self.config.resize_target_size[fw_cam][0] / raw_h,
            self.config.resize_target_size[fw_cam][1] / raw_w,
        )
        crop_offset = (
            self.config.crop_target_size[fw_cam][2],
            self.config.crop_target_size[fw_cam][3]
        ) # (h-offset, w-offset)

        projection_point = (
            raw_projection_point[0] * resize_scale[0] - crop_offset[0],
            raw_projection_point[1] * resize_scale[1] - crop_offset[1],
        )

        offset_h = torch.floor(projection_point[0] - self.config.fixed_proj_point[0] + 0.5).int()
        offset_w = torch.floor(projection_point[1] - self.config.fixed_proj_point[1] + 0.5).int()
        cur_offset_h, cur_offset_w = self.config.crop_target_size[fw_cam][2:]
        cur_offset_h = cur_offset_h + offset_h
        # print(f"cur_offset_h: {cur_offset_h}")
        cur_offset_h = torch.clip(cur_offset_h, min=0, max=self.config.resize_target_size[fw_cam][0]-self.config.crop_target_size[fw_cam][0])
        # print(f"clip cur_offset_h: {cur_offset_h}")
        cur_offset_w = cur_offset_w + offset_w
        # print(f"cur_offset_w: {cur_offset_w}")
        cur_offset_w = torch.clip(cur_offset_w, min=0, max=self.config.resize_target_size[fw_cam][1]-self.config.crop_target_size[fw_cam][1])
        # print(f"clip cur_offset_w: {cur_offset_w}")
        resize_target_size = copy.deepcopy(self.config.resize_target_size)
        crop_target_size = copy.deepcopy(self.config.crop_target_size)
        # crop_target_size[fw_cam][2:] = [cur_offset_h, cur_offset_w]
        crop_target_size[fw_cam] = [crop_target_size[fw_cam][0], crop_target_size[fw_cam][1], cur_offset_h, cur_offset_w]

        image = {}
        for cam_type, index in zip(self.config.multiview_cams, self.config.multiview_cams_index):
            single_resize_target_size = resize_target_size[cam_type]
            single_crop_target_size = crop_target_size[cam_type]
            img = resize_crop_image(pinhole_raw_img[index].float(), single_resize_target_size, single_crop_target_size) # 3xhxw uint8 -> float
            image[f"video:{cam_type}"] = img.unsqueeze(0).unsqueeze(0) / 127.5 - 1. # B, N, 3, H, W

        batch_size = image['video:/camera/front/main'].shape[0]
        input_dict = {
            "image": image, # dict, value: B, N, 3, H, W
            "batch_size": batch_size,
            "his_traj_next_waypoint": hist_action, # B, 4, 3
            "hist_img_feat": hist_img_feat,
            "subpath_polylines": subpath_polylines,
            "subpath_polylines_mask": subpath_polylines_mask,
            "navi_info": navi_info,
        }
    
        output = {}
        embeds_list = []
        for name in self.module_names:  # except traj
            if name != "traj":
                embeds, _, _, extra_outputs = getattr(self, f"{name}_encoder")(input_dict)
            else:
                embeds, extra_outputs = self.traj_encoder.ids_to_embed(
                    torch.tensor([self.traj_encoder.begin_meta_action_token_id], 
                                 dtype=torch.long, device=embeds_list[0].device).unsqueeze(0), # B, 1
                    is_begin_index=True,
                    input_dict=None,
                )
            if extra_outputs is not None and "pe" in extra_outputs:
                extra_outputs.pop("pe")
            embeds_list.append(embeds) # batch_size, seq_len, dim

            if extra_outputs is not None:
                output.update(extra_outputs)

        token_embeds = torch.cat(embeds_list, dim=1)
        position_embeds = torch.zeros_like(token_embeds)
        hist_img_feat = output["hist_img_feat"]

        return token_embeds, position_embeds, hist_img_feat

    def prefill(self, token_embeds, position_embeds):
        x = token_embeds

        for layer in self.transfomer["h"]:
            x = layer(x)
        hidden_states = x
        head_outputs = self.traj_head(hidden_states[:, -1:]) # B, 1, C
        logits = head_outputs["logits"] # B, 1, vocab_size
        return logits

    def decode(self, token_embeds, position_embeds, actions):
        # actions: , B, K, 2
        actions = actions[:, :, 0]
        inputs_embeds, extra_outputs = self.traj_encoder.ids_to_embed(
            actions, # B, K
            input_dict=None,
            keeplast=False,
        )
        if extra_outputs is not None and "pe" in extra_outputs:
            extra_outputs.pop("pe")

        x = torch.cat([token_embeds, inputs_embeds], dim=1)

        for layer in self.transfomer["h"]:
            x = layer(x)
        hidden_states = x
        head_outputs = self.traj_head(hidden_states[:, -1:]) # B, 1, C
        logits = head_outputs["logits"] # B, 1, vocab_size
        return logits
    @torch.no_grad()
    def planner_generate(
        self,
        image: torch.LongTensor = None,
        subpath_polylines: torch.FloatTensor = None,
        subpath_polylines_mask: torch.IntTensor = None,
        navi_info: torch.IntTensor = None,
        egos_gt_traj: Optional[torch.FloatTensor] = None,
        egos_pred_mask: Optional[torch.FloatTensor] = None,
        traj_next_waypoint: Optional[torch.FloatTensor] = None,
        his_egos_gt_traj: Optional[torch.FloatTensor] = None,
        egos_his_mask: Optional[torch.FloatTensor] = None,
        his_traj_next_waypoint: Optional[torch.FloatTensor] = None,
        hist_img_tokens: Optional[torch.LongTensor] = None,
        hist_img_feat: Optional[torch.LongTensor] = None,
        cam_extrinsics: Optional[torch.FloatTensor] = None,
        cali_path=None,
        projection_interface=None,
        meta_action_label=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        pred_times=1,
        rollout_num=1,
    ):
        if not self.config.use_kv_cache:
            raise RuntimeError("use_kv_cache should be True in generation!")

        batch_size = image['video:/camera/front/main'].shape[0] 
        input_dict = {
            "image": image, # dict, value: B, N, 3, H, W
            "batch_size": batch_size,
            "traj_next_waypoint": traj_next_waypoint,
            "egos_pred_mask": egos_pred_mask,
            "his_traj_next_waypoint": his_traj_next_waypoint,
            "his_egos_gt_traj": his_egos_gt_traj,
            "egos_his_mask": egos_his_mask,
            "hist_img_tokens": hist_img_tokens,
            "hist_img_feat": hist_img_feat,
            "subpath_polylines": subpath_polylines,
            "subpath_polylines_mask": subpath_polylines_mask,
            "navi_info": navi_info,
            "cam_extrinsics": cam_extrinsics,
            "cali_path": cali_path,
            "projection_interface": projection_interface,
            "meta_action_label": meta_action_label,
        }
    
        AR_traj_ids = None
        AR_traj_probs = None
        output = {}
        embeds_list = []
        for name in self.module_names[:-1]:  # except traj
            embeds, labels, masks, extra_outputs = getattr(self, f"{name}_encoder")(input_dict)
            if extra_outputs is not None and "pe" in extra_outputs:
                extra_outputs.pop("pe")

            embeds_list.append(
                embeds.unsqueeze(1).expand(-1, rollout_num, -1, -1).flatten(0, 1)
            ) # batch_size * rollout_num, seq_len, dim

            if extra_outputs is not None:
                output.update(extra_outputs)


        input_dict = {
            "cam_extrinsics": input_dict["cam_extrinsics"].unsqueeze(1).expand(-1, rollout_num, -1, -1, -1).flatten(0, 1), # B*rollout, cam, 4, 4
            "cali_path": [cali for cali in input_dict["cali_path"] for _ in range(rollout_num)],
            "projection_interface": [
                proj
                for proj in input_dict["projection_interface"]
                for _ in range(rollout_num)]
                if input_dict["projection_interface"] else input_dict["projection_interface"],
        }
        expand_batch_size = batch_size * rollout_num

        prefill_size = None
        for i in range(pred_times):
            if i == 0: # prefill
                embeds, extra_outputs = self.traj_encoder.ids_to_embed(
                    torch.tensor([self.traj_encoder.begin_meta_action_token_id], dtype=torch.long, device=embeds_list[0].device).unsqueeze(0).repeat(expand_batch_size, 1), # B, 1
                    is_begin_index=True,
                    input_dict=input_dict,
                )
                if extra_outputs is not None and "pe" in extra_outputs:
                    extra_outputs.pop("pe")

                embeds_sequence_size = [embeds.shape[1] for embeds in embeds_list] + [embeds.shape[1]]
                inputs_embeds = torch.cat(embeds_list + [embeds], dim=1)
                prefill_size = inputs_embeds.shape[1]
                input_pos = torch.arange(prefill_size, device=inputs_embeds.device)
            else:
                inputs_embeds, extra_outputs = self.traj_encoder.ids_to_embed(
                    AR_traj_ids, # B, K
                    input_dict=input_dict,
                )
                if extra_outputs is not None and "pe" in extra_outputs:
                    extra_outputs.pop("pe")
                input_pos = torch.arange(prefill_size + i - 1, prefill_size + i, device=inputs_embeds.device)

            x = inputs_embeds
            for layer in self.transfomer["h"]:
                x = layer(x, input_pos=input_pos)
            hidden_states = x

            # prefill
            if i == 0:
                hidden_states_list = torch.split(hidden_states, embeds_sequence_size, dim=1)
                for index, name in enumerate(self.module_names):
                    if name == "traj" or getattr(self, f"{name}_head") is None:
                        continue
                    single_hidden_state = hidden_states_list[index]
                    head_outputs = getattr(self, f"{name}_head")(single_hidden_state)
                    output.update(
                        {f"{name}_{key}": value for key, value in head_outputs.items()}
                    )

            head_outputs = self.traj_head(hidden_states[:, -1:]) # B, 1, C
            logits = head_outputs["logits"] # B, 1, vocab_size

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)  # b * pl * output_num
            if top_p is not None:
                probs = self.mask_top_p(probs, top_p=top_p)
            label_next = Categorical(probs).sample()  # b * pl
            if AR_traj_ids is None:
                AR_traj_ids = label_next
                AR_traj_probs = probs
            else:
                AR_traj_ids = torch.cat([AR_traj_ids, label_next], dim=1) # B, pred_times
                AR_traj_probs = torch.cat([AR_traj_probs, probs], dim=1) # B, pred_times, vocab_size

        meta_action_size = self.traj_encoder.meta_action_size
        action_ids = AR_traj_ids[:, -self.traj_encoder.main_action_length:]
        pred_egos_gt_traj, traj_actions = self.traj_encoder.decode(encode_tokens=action_ids)
        output["wp_outputs"] = pred_egos_gt_traj # B*rollout_num, N, 3
        output["traj_probs"] = AR_traj_probs # B*rollout_num, N, vocab_size
        output["traj_ids"] = action_ids # B*rollout_num, N
        output["meta_action_pred"] = AR_traj_ids[:, -(self.traj_encoder.main_action_length + meta_action_size):-self.traj_encoder.main_action_length]
        output["traj_actions"] = traj_actions  # B*rollout_num, N, 3(x, y, yaw)
        return output

    def mask_top_p(self, probs, top_p=0.9):
        """
        Mask probabilities in the tensor such that only the top probabilities
        that sum up to `top_p` are retained, and the rest are set to zero.

        Args:
            probs (torch.Tensor): An N x B torch tensor, where N is the number of samples
                                and B is the number of classes with classification probabilities.
            top_p (float): The cumulative probability threshold.

        Returns:
            torch.Tensor: A tensor with the same shape as `probs` but with masked probabilities.
        """
        # Sort probabilities in descending order for each sample
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calculate the cumulative sum along the last dimension
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask where the cumulative sum exceeds top_p
        mask = cumulative_probs > top_p
        
        # Shift mask right by one position to include at least one probability that contributes to top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0  # The first element should never be masked out
        
        # Zero out probabilities where the mask is True
        sorted_probs[mask] = 0.0
        
        # Restore the original order of probabilities
        masked_probs = torch.zeros_like(probs).scatter(dim=-1, index=sorted_indices, src=sorted_probs)
        
        return masked_probs
