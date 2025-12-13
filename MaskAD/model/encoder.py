import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from MaskAD.model.modules.global_attention import JointAttention
from MaskAD.model.utils import build_attn_bias_from_scene

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = config.agent_num + config.static_objects_num + config.lane_num

        self.agent_encoder = AgentFusionEncoder(
            time_len=config.past_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            tokens_mlp_dim=config.encoder_tokens_mlp_dim,
            channels_mlp_dim=config.encoder_channels_mlp_dim,
        )

        self.static_encoder = StaticFusionEncoder(
            dim=config.static_objects_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )

        self.lane_encoder = LaneFusionEncoder(
            lane_len=config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            tokens_mlp_dim=config.encoder_tokens_mlp_dim,
            channels_mlp_dim=config.encoder_channels_mlp_dim,
        )


        self.fusion_ecoder = JointAttention(
            dim_inputs = (config.hidden_dim, config.hidden_dim, config.hidden_dim),
            dim_head = config.encoder_dim_head,
            heads = config.encoder_num_heads,
            enable_attn_dist = config.enable_encoder_attn_dist,
            token_num = self.token_num,
            attend_kwargs = dict(dropout = config.encoder_attn_dropout),
        )

        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(5, config.hidden_dim)


    def forward(self, inputs):

        encoder_outputs = {}

        agents = inputs['agents_history']

        # static objects
        static = inputs['traffic_light_points']

        # vector maps
        lanes = inputs['polylines']

        agents_type = inputs['agents_type']
        # lanes_speed_limit = inputs['lanes_speed_limit']
        # lanes_has_speed_limit = inputs['lanes_has_speed_limit']


        B = agents.shape[0]

        encoding_agents, agents_mask, agent_pos = self.agent_encoder(agents,agents_type)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes)

        attn_dist = build_attn_bias_from_scene(
            agents,
            static,
            lanes,
            max_distance=None,
        )
        encoding_output = self.fusion_ecoder(
            (encoding_agents, encoding_static, encoding_lanes),
            (agents_mask, static_mask, lanes_mask),
            attn_dist,
        )
        # encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        encoding_pos = torch.cat([agent_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([agents_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts
        
        encoding_output = encoding_output + encoding_pos_result.view(B, self.token_num, -1)


           # 加上位置 embedding

        encoder_outputs['encoding'] = encoding_output
        encoder_outputs['encoding_agents'] = encoding_agents
        encoder_outputs['agents_mask'] = agents_mask
        encoder_outputs['static_mask'] = static_mask
        encoder_outputs['lanes_mask'] = lanes_mask

        return encoder_outputs



###### Scene encoder ######
class AgentFusionEncoder(nn.Module):
    def __init__(
        self,
        time_len=20,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
        num_agent_types=6,  # 0~5
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.STATE_DIM = 8  # x,y,yaw,vx,vy,length,width,height

        # 现在 agent_type 是 int(0~5)，用 Embedding
        self.type_emb = nn.Embedding(num_agent_types, channels_mlp_dim)

        # +1 是 valid flag（逻辑不变）
        self.channel_pre_project = Mlp(
            in_features=self.STATE_DIM + 1,  # 8 + 1
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x, agent_type):
        """
        x:         [B, P, V, 8]  (x, y, yaw, vx, vy, length, width, height)
        agent_type:[B, P]        int in {0,1,2,3,4,5}
        return:
          fused:  [B, P, hidden_dim]
          mask_p: [B, P]  True=padding agent
          pos:    [B, P, 7]  (保持原逻辑：先取最后帧的某些状态，再把最后3维改成“ego-like”标记)
        """

        B, P, V, D = x.shape
        assert D == self.STATE_DIM, f"expect x last dim={self.STATE_DIM}, got {D}"

        # ====== 1) 构造 pos（保持“步骤不变”）======
        # 原来 pos = x[:, :, -1, :7]，其中包含 cos/sin。
        # 现在输入是 yaw，所以这里用 cos(yaw), sin(yaw) 来对齐原语义。
        # pos 仍然是 7 维：x,y,cos,sin,vx,vy,width  （你也可以把 width 换成 length，取决于你后面怎么用）
        yaw = x[:, :, -1, 2]
        pos = torch.stack(
            [
                x[:, :, -1, 0],                 # x
                x[:, :, -1, 1],                 # y
                torch.cos(yaw),                 # cos(yaw)
                torch.sin(yaw),                 # sin(yaw)
                x[:, :, -1, 3],                 # vx              # width
            ],
            dim=-1,
        ).clone()  # [B,P,5]

        # 完全保留原来的“ego-like”写法（步骤不变）
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0

        # ====== 2) mask（逻辑不变）======
        mask_v = torch.sum(torch.ne(x, 0), dim=-1).to(x.device) == 0  # [B,P,V]
        mask_p = torch.sum(~mask_v, dim=-1) == 0                      # [B,P]

        # ====== 3) append valid flag（逻辑不变）======
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)   # [B,P,V,9]
        x = x.view(B * P, V, -1)                                      # [B*P,V,9]

        valid_indices = ~mask_p.view(-1)                              # [B*P]
        x = x[valid_indices]                                          # [N_valid,V,9]

        # ====== 4) Mixer（逻辑不变）======
        x = self.channel_pre_project(x)          # [N_valid,V,C]
        x = x.permute(0, 2, 1)                   # [N_valid,C,V]
        x = self.token_pre_project(x)            # [N_valid, tokens_mlp_dim, channels_mlp_dim] (取决于你 Mlp 实现)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        # pooling（逻辑不变）
        x = torch.mean(x, dim=1)                 # [N_valid, C]

        # ====== 5) type embedding（位置/步骤不变：pooling 后再加）======
        agent_type_flat = agent_type.view(B * P).to(x.device).long()
        agent_type_flat = agent_type_flat[valid_indices]              # [N_valid]
        type_embedding = self.type_emb(agent_type_flat)               # [N_valid, C]
        x = x + type_embedding

        # ====== 6) 输出回填（逻辑不变）======
        x = self.emb_project(self.norm(x))       # [N_valid, hidden_dim]

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)

class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, traffic_light_state)
        ''' 
        B, P, D = x.shape

        pos = torch.zeros((B, P, 5), device=x.device, dtype=x.dtype)
        # static: [0,1,0]
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :3], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1) 

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)

class LaneFusionEncoder(nn.Module):
    def __init__(
        self,
        lane_len,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
        num_tl_states=9,      # traffic_light_state: 0..8
        num_lane_types=20,    # lane_type: 0..19
    ):
        super().__init__()
        self._lane_len = lane_len
        self._channel = channels_mlp_dim
        self.num_tl_states = num_tl_states
        self.num_lane_types = num_lane_types

        geom_in_dim = 4  # (x, y, cos(h), sin(h))

        self.traffic_emb = nn.Embedding(num_tl_states, channels_mlp_dim)
        self.lane_type_emb = nn.Embedding(num_lane_types, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=geom_in_dim,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: [B, P, V, 5]
           (...,0:5) = (x, y, heading, traffic_light_state, lane_type)

        返回:
            lane_feat: [B, P, hidden_dim]
            mask_p:    [B, P]   (True: 整条 polyline 都是 padding)
            pos:       [B, P, 5]  (中点的 [x, y, heading, tl_norm, lane_type_norm])
        """
        B, P, V, D = x.shape
        assert D >= 5

        # -------- 1. 代表点 pos: 5 维 --------
        mid_idx = self._lane_len // 2
        mid_point = x[:, :, mid_idx, :]         # [B,P,5]
        mid_x = mid_point[..., 0]
        mid_y = mid_point[..., 1]
        mid_heading = mid_point[..., 2]
        mid_tl = mid_point[..., 3]
        mid_lane_type = mid_point[..., 4]

        pos = x.new_empty(B, P, 5)
        pos[..., 0] = mid_x
        pos[..., 1] = mid_y
        pos[..., 2] = mid_heading
        # 简单归一化到 [0,1] 区间（可选，不想归一化你可以直接用 mid_tl / mid_lane_type）
        pos[..., 3] = mid_tl / float(self.num_tl_states - 1)      # traffic_light_state_norm
        pos[..., 4] = mid_lane_type / float(self.num_lane_types - 1)  # lane_type_norm

        # -------- 2. 点级几何特征 --------
        coords = x[..., 0:2]          # [B,P,V,2]
        heading = x[..., 2]           # [B,P,V]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        geom = torch.cat([coords, cos_h.unsqueeze(-1), sin_h.unsqueeze(-1)], dim=-1)  # [B,P,V,4]

        # 与前面版本相同的 mask / embedding / Mixer 流程
        mask_v = torch.sum(torch.ne(geom, 0), dim=-1) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        geom = geom.view(B * P, V, -1)
        tl_state = x[..., 3].view(B * P, V).long()
        lane_type = x[..., 4].view(B * P, V).long()

        valid_indices = ~mask_p.view(-1)
        geom = geom[valid_indices]
        tl_state = tl_state[valid_indices]
        lane_type = lane_type[valid_indices]

        x_feat = self.channel_pre_project(geom)

        tl_state = torch.clamp(tl_state, 0, self.num_tl_states - 1)
        tl_emb = self.traffic_emb(tl_state)

        lane_type = torch.clamp(lane_type, 0, self.num_lane_types - 1)
        lane_emb = self.lane_type_emb(lane_type)

        x_feat = x_feat + tl_emb + lane_emb

        x_feat = x_feat.permute(0, 2, 1)
        x_feat = self.token_pre_project(x_feat)
        x_feat = x_feat.permute(0, 2, 1)
        for block in self.blocks:
            x_feat = block(x_feat)

        x_feat = torch.mean(x_feat, dim=1)
        x_feat = self.emb_project(self.norm(x_feat))      # [N_valid, hidden_dim]

        x_result = x.new_zeros(B * P, x_feat.shape[-1])
        x_result[valid_indices] = x_feat
        x_result = x_result.view(B, P, -1)

        return x_result, mask_p, pos




################### test ################

if __name__ == "__main__":
    import torch
    from types import SimpleNamespace

    # ===== 1. 配一个简单的 config =====
    config = SimpleNamespace()
    config.hidden_dim = 256

    config.agent_num = 64               # neighbor_agents_past 里 P 的大小
    config.static_objects_num = 16
    config.lane_num = 256

    config.static_objects_dim = 3

    config.past_len = 11              # AgentFusionEncoder: time_len
    config.lane_len = 30               # LaneFusionEncoder: lane_len

    config.encoder_drop_path_rate = 0.1
    config.encoder_depth = 2
    config.encoder_tokens_mlp_dim = 64
    config.encoder_channels_mlp_dim = 128

    config.encoder_dim_head = 64
    config.encoder_num_heads = 4
    config.enable_encoder_attn_dist = True
    config.encoder_attn_dropout = 0.1

    # ===== 2. 构造随机输入 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    N_agents = config.agent_num
    N_static = config.static_objects_num
    N_lanes = config.lane_num
    T_hist = config.past_len
    lane_len = config.lane_len

    # neighbor_agents_past: (x, y, cos, sin, vx, vy, w, l, type(3)) → D = 11
    agents_history = torch.randn(B, N_agents, T_hist,8, device=device)

    # static_objects: (x, y, cos, sin, w, l, type(4)) → D = 10
    traffic_light_points = torch.randn(B, N_static, 3, device=device)

    # lanes: (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4)) → D = 12
    polylines = torch.randn(B, N_lanes, lane_len, 5, device=device)

    agents_type = torch.randint(0, 4, (B, N_agents), device=device)

    inputs = {
        "agents_history": agents_history,
        "traffic_light_points": traffic_light_points,
        "polylines": polylines,
        "agents_type": agents_type,
    }
    for k, v in inputs.items():
        print(k, v.shape)
    # ===== 3. 实例化 Encoder 并前向 =====
    encoder = Encoder(config).to(device)

    with torch.no_grad():
        outputs = encoder(inputs)

    # ===== 4. 打印检查一下 =====
    encoding = outputs["encoding"]
    print("encoding shape:", encoding.shape)  # 期望: [B, token_num, hidden_dim]

    print("Test passed.")
