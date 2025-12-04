import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from MaskAD.model.tools.global_attention import JointAttention
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
            dim=10,
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
        self.pos_emb = nn.Linear(7, config.hidden_dim)


    def forward(self, inputs):

        encoder_outputs = {}

        neighbors = inputs['neighbor_agents_past']

        # static objects
        static = inputs['static_objects']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']


        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbor_pos = self.agent_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        attn_dist = build_attn_bias_from_scene(
            neighbors,
            static,
            lanes,
            max_distance=None,
        )
        encoding_output = self.fusion_ecoder(
            (encoding_neighbors, encoding_static, encoding_lanes),
            (neighbors_mask, static_mask, lanes_mask),
            attn_dist,
        )
        # encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim=1)

        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_output = encoding_output + encoding_pos_result.view(B, self.token_num, -1)


           # 加上位置 embedding
        encoding_output = encoding_output + encoding_pos_result.view(B, self.token_num, -1)

        encoder_outputs['encoding'] = encoding_output
        encoder_outputs['neighbors_mask'] = neighbors_mask
        encoder_outputs['static_mask'] = static_mask
        encoder_outputs['lanes_mask'] = lanes_mask

        return encoder_outputs



###### Scene encoder ######
class AgentFusionEncoder(nn.Module):
    def __init__(self, time_len=20, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)


    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        '''
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :7].clone() # x, y, cos, sin
        # neighbor: [1,0,0]
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0 # B, P, 7
        
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1) # [B, P, V, 9]

        valid_indices = ~mask_p.view(-1)  # [B*P]
        x = x[valid_indices]  # x: [N_valid, V, 9]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)  # # x: [N_valid, tokens_mlp_dim, channels_mlp_dim]
        for block in self.blocks:
            x = block(x)  

        # pooling
        x = torch.mean(x, dim=1) # x: [N_valid, channels_mlp_dim]

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        ''' 
        B, P, _ = x.shape

        pos = x[:, :, :7].clone() # x, y, cos, sin
        # static: [0,1,0]
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1) 

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)

class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._lane_len = lane_len
        
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].clone() # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0,0,1]
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)  # [B*P] 的维度
        x = x[valid_indices]  

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)  

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data


        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class RouteEncoder(nn.Module):
    def __init__(self, route_num, route_points_num, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * route_points_num, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
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

####### KV encoder #################
class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [globalselfattention(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)



################### test ################




if __name__ == "__main__":
    import torch
    from types import SimpleNamespace

    # ===== 1. 配一个简单的 config =====
    config = SimpleNamespace()
    config.hidden_dim = 192

    config.agent_num = 16               # neighbor_agents_past 里 P 的大小
    config.static_objects_num = 32
    config.lane_num = 64

    config.past_len = 20               # AgentFusionEncoder: time_len
    config.lane_len = 20               # LaneFusionEncoder: lane_len

    config.encoder_drop_path_rate = 0.1
    config.encoder_depth = 2
    config.encoder_tokens_mlp_dim = 64
    config.encoder_channels_mlp_dim = 128

    config.encoder_dim_head = 48
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
    neighbor_agents_past = torch.randn(B, N_agents, T_hist, 11, device=device)

    # static_objects: (x, y, cos, sin, w, l, type(4)) → D = 10
    static_objects = torch.randn(B, N_static, 10, device=device)

    # lanes: (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4)) → D = 12
    lanes = torch.randn(B, N_lanes, lane_len, 12, device=device)

    # 车道限速信息
    lanes_speed_limit = torch.randn(B, N_lanes, 1, device=device)
    lanes_has_speed_limit = torch.randint(0, 2, (B, N_lanes, 1), device=device).bool()

    # routes 目前 Encoder.forward 没用到，造一个占位
    routes = torch.randn(B, 4, 10, 4, device=device)  # 只是举例，shape 随便

    inputs = {
        "neighbor_agents_past": neighbor_agents_past,
        "static_objects": static_objects,
        "lanes": lanes,
        "lanes_speed_limit": lanes_speed_limit,
        "lanes_has_speed_limit": lanes_has_speed_limit,
        "routes": routes,
    }

    # ===== 3. 实例化 Encoder 并前向 =====
    encoder = Encoder(config).to(device)

    with torch.no_grad():
        outputs = encoder(inputs)

    # ===== 4. 打印检查一下 =====
    encoding = outputs["encoding"]
    print("encoding shape:", encoding.shape)  # 期望: [B, token_num, hidden_dim]
    print("neighbors_mask shape:", outputs["neighbors_mask"].shape)
    print("static_mask shape:", outputs["static_mask"].shape)
    print("lanes_mask shape:", outputs["lanes_mask"].shape)

    print("Test passed.")
