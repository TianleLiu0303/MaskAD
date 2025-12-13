import math
import torch
import torch.nn as nn
from timm.layers import Mlp
from timm.layers import DropPath

# from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
# from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
# from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
# from diffusion_planner.model.module.mixer import MixerBlock
from MaskAD.model.modules.dit import TimestepEmbedder, DiTBlock, FinalLayer
from MaskAD.model.encoder import MixerBlock


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len

        self.dit = DiT(
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= (config.future_len+1) * 4, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
        )
    def forward(self, encoder_outputs, inputs):
        agents_history = inputs["agents_history"]  
        mask_v = torch.sum(torch.ne(agents_history[..., :2], 0), dim=-1) == 0   # [B,P,V]
        neighbor_current_mask = mask_v[:, :, -1]                                # [B,P]
        inputs["neighbor_current_mask"] = neighbor_current_mask[:, 1:]

        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        mask_emb = encoder_outputs['mask_emb'] # [B, N, T, D_model]

        traj = inputs['sampled_trajectories']  
        # 拆开当前 + 未来
        traj_current = traj[:, :, :1, :]    # [B, P, 1, 4]
        traj_future  = traj[:, :, 1:, :]    # [B, P, T, 4]

        # 只对未来部分加 mask
        traj_future = traj_future + mask_emb
        B,P, _, _ =  traj.shape
        # 再拼回去
        sampled_trajectories = torch.cat([traj_current, traj_future], dim=2)  # [B, P, 1+T, 4]
        sampled_trajectories = sampled_trajectories.reshape(B, P, -1)
        # sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
        diffusion_time = inputs['diffusion_time']

        return {
                "x_start":self.dit(
                    sampled_trajectories, 
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask[:,1:]
                ).reshape(B, P, -1, 4)
        }




class DiT(nn.Module):
    def __init__(self, route_encoder: nn.Module, depth, output_dim, hidden_dim=256, heads=6, dropout=0.1, mlp_ratio=4.0):

        '''
        初始化的参数：
                route_encoder: 导航信息编码器
                depth: 几层DiTBlock
                output_dim: 输出维度
                hidden_dim: 隐藏层维度
                heads: 多头注意力的头数 
                dropout: dropout概率
                mlp_ratio: MLP隐藏层和dim的比率
        '''
        super().__init__()
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        # 引入的解码decoder中代码块
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
               
    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        B, P, _ = x.shape # 

        x = self.preproj(x) # 维度为[B, 33, 256]

        navigation_encoding = self.route_encoder(route_lanes) # 维度是（B, 256）
        y = navigation_encoding

        ##### 一下为调试代码 #####
        y = y + self.t_embedder(t)  # # 维度是（B, 256）
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:,1:] = neighbor_current_mask 
        
        for block in self.blocks:
            '''
        输入的维度：
              x: 主序列输入张量，形状为 (B, 32， 256)
              cross_c: 交叉注意力条件张量，形状为 (B, 336, 256)
              y: 条件张量，形状为 (B, 256)
              attn_mask: 注意力掩码，形状为 (B, 33)
        输出:
              x: 形状为 (B, 32, 256)
            '''
            x = block(x, cross_c, y, attn_mask)   # 维度是 [B, 32, 256]
        
        x = self.final_layer(x, y) 
        
        return x


class RouteEncoder(nn.Module):
    def __init__(
        self,
        route_num,
        lane_len,
        drop_path_rate=0.3,
        hidden_dim=256,
        tokens_mlp_dim=33,
        channels_mlp_dim=64,
        num_tl_states=9,      # traffic_light_state: 0..8
        num_lane_types=20,    # lane_type: 0..19
    ):
        super().__init__()

        self._channel = channels_mlp_dim
        self.route_num = route_num
        self.lane_len = lane_len
        self.num_tl_states = num_tl_states
        self.num_lane_types = num_lane_types

        # 4 维几何特征: (x, y, cos(h), sin(h))
        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # 整个 route 视作一个序列，长度 = route_num * lane_len
        self.token_pre_project = Mlp(
            in_features=route_num * lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # traffic light / lane type 嵌入
        self.traffic_emb = nn.Embedding(num_tl_states, channels_mlp_dim)
        self.lane_type_emb = nn.Embedding(num_lane_types, channels_mlp_dim)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

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
            route_feat: [B, hidden_dim]   每个场景一个 navigation 向量
        """
        B, P, V, D = x.shape

        # ---- 1. 几何 + 语义分解 ----
        coords = x[..., 0:2]          # [B,P,V,2]
        heading = x[..., 2]           # [B,P,V]
        tl_state = x[..., 3].long()   # [B,P,V]
        lane_type = x[..., 4].long()  # [B,P,V]

        # 几何特征: (x, y, cos(h), sin(h))
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        geom = torch.cat(
            [coords, cos_h.unsqueeze(-1), sin_h.unsqueeze(-1)],
            dim=-1
        )                              # [B,P,V,4]

        # ---- 2. mask：全 0 视为 padding route ----
        # 这里对 geom 判断是否全 0，当一个场景所有 route 都是 0 时，认为该 batch 无效
        mask_v = torch.sum(torch.ne(geom, 0.0), dim=-1) == 0     # [B,P,V]
        mask_b = torch.sum(~mask_v.view(B, -1), dim=-1) == 0     # [B] 每个场景是否“全 padding”

        # ---- 3. 展平成一个长序列: B, (P*V), 4 ----
        geom = geom.view(B, P * V, 4)
        tl_state = tl_state.view(B, P * V)
        lane_type = lane_type.view(B, P * V)

        valid_indices = ~mask_b     # [B]
        geom = geom[valid_indices]  # [B_valid, P*V, 4]
        tl_state = tl_state[valid_indices]
        lane_type = lane_type[valid_indices]

        # ---- 4. 通道投影 + 语义 embedding ----
        x_feat = self.channel_pre_project(geom)        # [B_valid, P*V, C]

        # clamp 防止越界
        tl_state = torch.clamp(tl_state, 0, self.num_tl_states - 1)
        lane_type = torch.clamp(lane_type, 0, self.num_lane_types - 1)

        tl_emb = self.traffic_emb(tl_state)            # [B_valid, P*V, C]
        lane_emb = self.lane_type_emb(lane_type)       # [B_valid, P*V, C]

        x_feat = x_feat + tl_emb + lane_emb            # [B_valid, P*V, C]

        # ---- 5. token 维度 Mixer ----
        # 现在视为 [B_valid, C, N_token]，N_token = P*V
        x_feat = x_feat.permute(0, 2, 1)               # [B_valid, C, P*V]
        x_feat = self.token_pre_project(x_feat)        # [B_valid, tokens_mlp_dim, C]
        x_feat = x_feat.permute(0, 2, 1)               # [B_valid, C, tokens_mlp_dim]

        x_feat = self.Mixer(x_feat)                    # [B_valid, tokens_mlp_dim, C]

        # 沿着 token 维度做 average pooling
        x_feat = torch.mean(x_feat, dim=1)             # [B_valid, C]

        # ---- 6. 最终投影到 hidden_dim ----
        x_feat = self.emb_project(self.norm(x_feat))    # [B_valid, hidden_dim]

        # ---- 7. 回填到 B 个场景 ----
        out = x.new_zeros(B, x_feat.shape[-1])
        out[valid_indices] = x_feat                    # [B, hidden_dim]

        return out




###############  test #####################
def main():
    from types import SimpleNamespace

    # ===== 1. 配一个简单的 config =====
    config = SimpleNamespace()
    config.hidden_dim = 256

    config.agent_num = 64               # neighbor_agents_past 里 P 的大小
    config.static_objects_num = 16
    config.lane_num = 256
    config.predicted_neighbor_num = 63

    config.static_objects_dim = 3

    config.past_len = 11              # AgentFusionEncoder: time_len
    config.lane_len = 30               # LaneFusionEncoder: lane_len
    config.future_len = 80

    config.encoder_drop_path_rate = 0.1
    config.encoder_depth = 2
    config.encoder_tokens_mlp_dim = 64
    config.encoder_channels_mlp_dim = 128

    config.encoder_dim_head = 64
    config.encoder_num_heads = 4
    config.enable_encoder_attn_dist = True
    config.encoder_attn_dropout = 0.1

    config.decoder_drop_path_rate = 0.1
    config.encoder_drop_path_rate = 0.1
    config.hidden_dim = 256
    config.num_heads = 4
    config.future_len = 80                  # 未来 40 帧 -> (40 + 1) * 4 维度
    config.route_num = 16                   # 对应 P
    config.lane_len = 64                    # 对应 V
    config.decoder_depth = 1          # DiTBlock 层数
    # ===== 2. 构造随机输入 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    P = 1 + config.predicted_neighbor_num  # 33
    V = config.lane_len                   # 64
    D_route = 4  
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

    route_lanes = torch.randn(B, 16, lane_len, 5, device=device)

    sampled_trajectories = torch.randn(
        B, P, (config.future_len+1), 4,
        device=device
    )

    # diffusion_time: [B]
    diffusion_time = torch.randint(
        low=0, high=1000, size=(B,), device=device
    ).float()

    encoder_outputs = {
        "encoding": torch.randn(B, 336, config.hidden_dim, device=device),
         "mask_emb": torch.randn(B, P, (config.future_len), 4, device=device),
    }
    inputs = {
        "agents_history": agents_history,
        "traffic_light_points": traffic_light_points,
        "polylines": polylines,
        "agents_type": agents_type,
        "route_lanes": route_lanes,
        "diffusion_time": diffusion_time,
        "sampled_trajectories": sampled_trajectories,     
    }

    # ========= 4. 前向测试 =========
    decoder = Decoder(config).to(device)
    decoder.eval()
    with torch.no_grad():
        outputs = decoder(encoder_outputs, inputs)

    x_start = outputs["x_start"]
    print("x_start shape:", x_start.shape)

if __name__ == "__main__":
    main()
