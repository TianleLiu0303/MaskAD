import math
import torch
import torch.nn as nn
from timm.layers import Mlp

def modulate(x, shift, scale, only_first=False):
    """
    调整输入张量 x 的值，通过 shift 和 scale 参数进行调制。
    
    输入:
    - x: 输入张量，形状为 (B, P, D)，其中 B 是批量大小，P 是序列长度，D 是特征维度。
    - shift: 调整偏移量张量，形状为 (B, D)。
    - scale: 调整缩放因子张量，形状为 (B, D)。
    - only_first: 布尔值，是否仅对序列的第一个元素进行调制。

    输出:
    - 调制后的张量，形状与输入 x 相同。
    """
    if only_first: #当 only_first=True 时，只对序列中的第一个 token 应用 scale/shift，其余 token 保持不变
        x_first, x_rest = x[:, :1], x[:, 1:]  # 分离第一个元素和剩余部分
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


def scale(x, scale, only_first=False):
    """
    对输入张量 x 的值进行缩放，通过 scale 参数调整。modulate的简化版本，只进行缩放，不进行偏移。

    输入:
    - x: 输入张量，形状为 (B, P, D)，其中 B 是批量大小，P 是序列长度，D 是特征维度。
    - scale: 缩放因子张量，形状为 (B, D)。
    - only_first: 布尔值，是否仅对序列的第一个元素进行缩放。

    输出:
    - 缩放后的张量，形状与输入 x 相同。
    """
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]  # 分离第一个元素和剩余部分
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    将标量时间步嵌入到向量表示中。映射到更高的维度空间中，以便与其他嵌入进行结合。

    输入:
    - t: 一个 1-D 张量，形状为 (N,)，表示每个批次元素的时间步索引，可以是小数。
    
    输出:
    - t_emb: 一个 2-D 张量，形状为 (N, hidden_size)，表示时间步的嵌入表示。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        初始化 TimestepEmbedder 模块。

        输入:
        - hidden_size: 输出嵌入的维度大小。
        - frequency_embedding_size: 频率嵌入的维度大小，默认为 256。
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入。

        输入:
        - t: 一个 1-D 张量，形状为 (N,)，表示每个批次元素的时间步索引，可以是小数。
        - dim: 输出嵌入的维度大小。
        - max_period: 控制嵌入的最小频率，默认为 10000。

        输出:
        - embedding: 一个 2-D 张量，形状为 (N, dim)，表示时间步的正弦嵌入。
        """
        # 计算频率
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # 计算正弦和余弦嵌入
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果维度是奇数，补零
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        前向传播，生成时间步的嵌入表示。

        输入:
        - t: 一个 1-D 张量，形状为 (N,)，表示每个批次元素的时间步索引。

        输出:
        - t_emb: 一个 2-D 张量，形状为 (N, hidden_size)，表示时间步的嵌入表示。
        """
        # 生成频率嵌入
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 通过 MLP 生成最终嵌入
        t_emb = self.mlp(t_freq)
        return t_emb


'''
当使用这个模块时候需要：
        输入的维度：
              x: 主序列输入张量，形状为 (B, 32， 256)
              cross_c: 交叉注意力条件张量，形状为 (B, 336, 256)
              y: 条件张量，形状为 (B, 256)
              attn_mask: 注意力掩码，形状为 (B, 32)
        输出:
              x: 形状为 (B, 32, 256)
'''
class DiTBlock(nn.Module):
    """
    最小的DiT 块，带有自适应层归一化零（adaLN-Zero）条件，用于自注意力和交叉注意力。
    构建参数：
        - dim: 输入张量的特征维度大小。`
        - heads: 多头自注意力的头数。
        - dropout: dropout 概率。
        - mlp_ratio: MLP 隐藏层和dim的比率。

    """
    def __init__(self, dim=256, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # 第一层归一化
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)  # 多头自注意力
        self.norm2 = nn.LayerNorm(dim)  # 第二层归一化
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP 隐藏层维度
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # GELU 激活函数
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)  # 第一层 MLP
        # 用于生成 shift/scale/gate 参数的自适应调制层
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # SiLU 激活
            nn.Linear(dim, 6 * dim, bias=True)  # 输出 6 * dim，用于 shift/scale/gate
        )
        # 
        self.norm3 = nn.LayerNorm(dim)  # 第三层归一化
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)  # 多头交叉注意力
        self.norm4 = nn.LayerNorm(dim)  # 第四层归一化
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)  # 第二层 MLP

    def forward(self, x, cross_c, y, attn_mask):
        '''输入的维度：
              x: 主序列输入张量，形状为 (B, 32， 256)
              cross_c: 交叉注意力条件张量，形状为 (B, 336, 256)
              y: 条件张量，形状为 (B, 256)
              attn_mask: 注意力掩码，形状为 (B, 32)
        输出:
              x: 形状为 (B, 32, 256)
        '''
        # 通过 adaLN_modulation 生成 shift/scale/gate 参数 维度为 (B, 6 * D)，每一个为 (B, 256)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

        # 对 x 进行归一化和调制
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # gate_msa 控制自注意力输出的权重
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]

        # 第二次归一化和调制
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)

        # gate_mlp 控制 MLP 输出的权重
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        # 交叉注意力，cross_c 是条件张量，就是场景条件用于生成K，V
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]

        # 最后一次归一化和 MLP
        x = self.mlp2(self.norm4(x))

        return x
    
    
class FinalLayer(nn.Module):

    def __init__(self, hidden_size, output_size):
        """
        初始化 FinalLayer 模块。

        输入:
        - hidden_size: 输入张量的特征维度大小。
        - output_size: 输出张量的特征维度大小。
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)  # 最终的 LayerNorm 层
        self.proj = nn.Sequential(  # 投影层，用于将输入映射到输出维度
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(  # 生成scale和shift参数的自适应调制层
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        """
        前向传播，生成最终的输出。

        输入:
        - x: 输入张量，形状为 (B,32,256)。
        - y: 条件张量，形状为 (B, 256)。

        输出:
        - 输出张量，形状为 (B, P, output_size)。
        """
        B, P, _ = x.shape
        
        # 通过 adaLN_modulation 生成 shift 和 scale 参数
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        
        # 调制输入张量
        x = modulate(self.norm_final(x), shift, scale)
        
        # 投影到输出维度
        x = self.proj(x)
        return x