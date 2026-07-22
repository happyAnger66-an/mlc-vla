import torch
from torch import nn

from model.encoder.polyline_encoder import PointNetPolylineEncoder
from model.data_define import navi_cls_mapping


class NaviTokenizer(nn.Module):
    def __init__(self, n_embed, navi_cls1_vocab_size=7, navi_cls2_vocab_size=62, 
                 navi_cls3_vocab_size=33, navi_cls4_vocab_size=22):
        super(NaviTokenizer, self).__init__()
        self.polyline_encoder = PointNetPolylineEncoder(2, n_embed)
        self.navi_embedding_1 = nn.Embedding(navi_cls1_vocab_size, n_embed, padding_idx=0)
        self.navi_embedding_2 = nn.Embedding(navi_cls2_vocab_size, n_embed, padding_idx=0)
        self.navi_embedding_3 = nn.Embedding(navi_cls3_vocab_size, n_embed, padding_idx=0)
        self.navi_embedding_4 = nn.Embedding(navi_cls4_vocab_size, n_embed, padding_idx=0)
        # lane navi_info embeding
        self.lane_direction_embeddings = nn.ModuleList([nn.Embedding(2, n_embed) for _ in range(5)])
        self.lane_ok_embeddings = nn.Embedding(2, n_embed)

        self.apply(self._init_weights)

        # print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.navi_embedding_1.weight.numel()
            n_params -= self.navi_embedding_2.weight.numel()
            n_params -= self.navi_embedding_3.weight.numel()
            n_params -= self.navi_embedding_4.weight.numel()
            n_params -= self.lane_ok_embeddings.weight.numel()
            for module in self.lane_direction_embeddings:
                n_params -= module.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_dict):
        """
        Args:
            subpath_polylines: [batch_size, subpath_seq_len, 4, 2]
            subpath_polylines_mask: [batch_size, subpath_seq_len]
            navi_info: [batch_size, navi_seq_len, 5]  (cls1, cls2, cls3, cls4, remain_distance)
        """
        subpath_polylines = input_dict["subpath_polylines"]
        subpath_polylines_mask = input_dict["subpath_polylines_mask"]
        navi_info = input_dict["navi_info"]

        polyline_embeding = self.polyline_embedding(subpath_polylines, subpath_polylines_mask)
        navi_info_embeding = self.navi_embedding(navi_info)

        x = torch.cat((polyline_embeding, navi_info_embeding), dim=1)  # B x N x d

        labels = input_dict.get("navi_label", None)
        return x, labels, None, None

    def polyline_embedding(self, polylines, polylinemask):
        b, n, polyline_len, _ = polylines.shape
        # convert B x N mask (each element indicate True num of points)
        # to B x N x polyline_len x 2 mask
        polylinemask_full = torch.arange(polyline_len, device=polylines.device)
        polylinemask_full = polylinemask_full.view(1, 1, polyline_len).expand(b, n, polyline_len)
        polylinemask_full = polylinemask_full < polylinemask.unsqueeze(-1)

        polyline_embeding = self.polyline_encoder(polylines, polylinemask_full)  # B x N x d
        polyline_embeding = polyline_embeding * (polylinemask.unsqueeze(-1) > 0)  # set masked embedding to 0
        return polyline_embeding

    def navi_embedding(self, navi_info):        
        cls1_embeding = self.navi_embedding_1(navi_info[:, :, 0])
        cls2_embeding = self.navi_embedding_2(navi_info[:, :, 1])
        cls3_embeding = self.navi_embedding_3(navi_info[:, :, 2])
        cls4_embeding = self.navi_embedding_4(navi_info[:, :, 3])
        lane_info_embeding = self.lane_info_embedding(navi_info)

        navi_info_embeding = cls1_embeding + cls2_embeding + cls3_embeding + cls4_embeding + lane_info_embeding
        return navi_info_embeding
    
    def lane_info_embedding(self, navi_info):
        direction_index = ((navi_info[:, :, 0] == navi_cls_mapping['lane_info']) 
                      | (navi_info[:, :, 0] == navi_cls_mapping['lane_highline_info'])
                      | (navi_info[:, :, 0] == navi_cls_mapping['tld']))
        lane_index = ((navi_info[:, :, 0] == navi_cls_mapping['lane_info']) 
                      | (navi_info[:, :, 0] == navi_cls_mapping['lane_highline_info']))
        # 非车道token没有lane_is_ok和右掉头信息.
        flags = navi_info[:, :, 4]

        # 代替 flags & 0x1
        lane_is_ok_flag = torch.remainder(flags, 2)
        lane_is_ok = self.lane_ok_embeddings(lane_is_ok_flag) * lane_index.unsqueeze(2)

        # 代替 (flags >> 1) & 0x1
        lane_direction_rightturn_flag = torch.remainder(torch.div(flags, 2, rounding_mode='floor'), 2)
        lane_direction_rightturn = self.lane_direction_embeddings[0](lane_direction_rightturn_flag) * lane_index.unsqueeze(2)
        
        # 代替 (flags >> 2) & 0x1
        lane_direction_right_flag = torch.remainder(torch.div(flags, 4, rounding_mode='floor'), 2)
        lane_direction_right = self.lane_direction_embeddings[1](lane_direction_right_flag)
        
        # 代替 (flags >> 3) & 0x1
        lane_direction_straight_flag = torch.remainder(torch.div(flags, 8, rounding_mode='floor'), 2)
        lane_direction_straight = self.lane_direction_embeddings[2](lane_direction_straight_flag)
        
        # 代替 (flags >> 4) & 0x1
        lane_direction_left_flag = torch.remainder(torch.div(flags, 16, rounding_mode='floor'), 2)
        lane_direction_left= self.lane_direction_embeddings[3](lane_direction_left_flag)
        
        # 代替 (flags >> 5) & 0x1
        lane_direction_leftturn_flag = torch.remainder(torch.div(flags, 32, rounding_mode='floor'), 2)
        lane_direction_leftturn= self.lane_direction_embeddings[4](lane_direction_leftturn_flag)
        
        lane_embeding = lane_is_ok + lane_direction_rightturn + lane_direction_right + lane_direction_straight + \
                        lane_direction_left + lane_direction_leftturn
        lane_embeding = lane_embeding * direction_index.unsqueeze(2)
        return lane_embeding
    

if __name__ == "__main__":
    # 大类：pad, tld, spd_limit, action_1, action_2, lane_info, lane_highline_info
    navi_cls1_vocab_size = 7
    # cls2：pad, 15个lane_line type，5个tld color(k, r, y, g, gray), 13个spd_limit [0-12], 13个lane_id, 
    # 15个主action
    navi_cls2_vocab_size = 62
    # cls3: pad, 9个lane_line color, 4个lane type(无效车道，普通车道，公交车道，潮汐车道叉号), 19个辅助action,
    navi_cls3_vocab_size = 33
    # cls4: pad, 21个remain_distance
    navi_cls4_vocab_size = 22


    tokenizer = NaviTokenizer(768, navi_cls1_vocab_size, navi_cls2_vocab_size, navi_cls3_vocab_size, navi_cls4_vocab_size)
    subpath_polylines = torch.randn(2, 30, 4, 2)
    subpath_polylines_mask = torch.randint(0, 1, (2, 30))
    navi_info = torch.randint(0, 7, (2, 33, 5))

    input_dict = dict()
    input_dict["subpath_polylines"] = subpath_polylines
    input_dict["subpath_polylines_mask"] = subpath_polylines_mask
    input_dict["navi_info"] = navi_info

    out = tokenizer(input_dict)
    print(out[0].shape)
