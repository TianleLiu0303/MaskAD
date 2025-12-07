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
            output_dim= (config.future_len) * 4, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
        )
    def forward(self, encoder_outputs, inputs):
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["agents_past"][:, 1:, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        mask_emb = encoder_outputs['mask_emb'] # [B, N, T, D_model]

        sampled_trajectories = inputs['sampled_trajectories'] + mask_emb # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
        sampled_trajectories = sampled_trajectories.reshape(B, P, -1)
        # sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
        diffusion_time = inputs['diffusion_time']

        return {
                "x_start":self.dit(
                    sampled_trajectories, 
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask
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
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=33, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)



###############  test #####################
def main():
    # ========= 1. 构造一个假的 config =========
    class Config:
        pass

    config = Config()
    config.decoder_drop_path_rate = 0.1
    config.encoder_drop_path_rate = 0.1
    config.hidden_dim = 256
    config.num_heads = 4

    config.predicted_neighbor_num = 32      # P = 1(ego) + 31 = 32
    config.future_len = 40                  # 未来 40 帧 -> (40 + 1) * 4 维度
    config.route_num = 16                   # 对应 P
    config.lane_len = 64                    # 对应 V
    config.decoder_depth = 1          # DiTBlock 层数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= 2. 构造 Decoder =========
    decoder = Decoder(config).to(device)
    decoder.eval()

    # ========= 3. 构造假输入数据 =========
    B = 2  # batch size
    P = 1 + config.predicted_neighbor_num  # 33
    V = config.lane_len                   # 64
    D_route = 4                           # route_lanes 的最后维度至少 4（你前面只用到 :4）

    # ego 当前状态: [B, 4]
    ego_current_state = torch.randn(B, 4, device=device)

    # neighbor 过去轨迹: [B, P-1, T_past, 4]
    T_past = 20
    neighbor_agents_past = torch.randn(B, P - 1, T_past, 4, device=device)

    # route_lanes: [B, P, V, D_route]
    route_lanes = torch.randn(B, 16, V, D_route, device=device)

    # sampled_trajectories: [B, P, (future_len + 1) * 4]
    sampled_trajectories = torch.randn(
        B, P, config.future_len  * 3,
        device=device
    )

    # diffusion_time: [B]
    diffusion_time = torch.randint(
        low=0, high=1000, size=(B,), device=device
    ).float()

    # encoder_outputs['encoding']: [B, N, D]
    # 比如 N = 336（你注释里写的），D = hidden_dim
    N = 336
    encoder_outputs = {
        "encoding": torch.randn(B, N, config.hidden_dim, device=device)
    }

    # 组装 inputs 字典，键名要和 Decoder.forward 里的一致
    inputs = {
        "ego_current_state": ego_current_state,
        "neighbor_agents_past": neighbor_agents_past,
        "route_lanes": route_lanes,
        "sampled_trajectories": sampled_trajectories,
        "diffusion_time": diffusion_time,
    }

    # ========= 4. 前向测试 =========
    with torch.no_grad():
        outputs = decoder(encoder_outputs, inputs)

    x_start = outputs["x_start"]
    print("x_start shape:", x_start.shape)

if __name__ == "__main__":
    main()
