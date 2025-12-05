import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskNet(nn.Module):
    """
    输入:
        scene_feat: [B, N, D]
    输出:
        mask:      [B, N, T]
        mask_emb:  [B, N, T, D_model]
    """

    def __init__(
        self,
        hidden_dim,        # D, scene_feat hidden dim
        T_fut,             # 未来预测步数 T
        d_model,           # 加入到 DiT 的 embedding 维度
        time_emb_dim=64,
        num_heads=4,
        mlp_hidden=128,
    ):
        super().__init__()

        self.T = T_fut

        # ---- 1. 时间 embedding ----
        self.time_emb = nn.Embedding(T_fut, time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)  # match scene dim

        # ---- 2. cross-attention: 时间查询 → 场景token ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # ---- 3. MLP: 每个 token × time 输出一个 scalar ----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

        # ---- 4. mask embedding ----
        self.mask_emb_proj = nn.Linear(1, d_model)
        self.emb_scale = nn.Parameter(torch.tensor(1.0))


    def forward(self, scene_feat):
        B, N, D = scene_feat.shape

        # scene tokens = K, V
        K = scene_feat
        V = scene_feat

        # ---- Step 1: time embeddings → queries ----
        t_ids = torch.arange(self.T, device=scene_feat.device)  # [T]
        t_emb = self.time_emb(t_ids)                            # [T, time_emb_dim]
        t_emb = self.time_proj(t_emb)                           # [T, D]
        Q = t_emb.unsqueeze(0).expand(B, self.T, D)             # [B, T, D]

        # ---- Step 2: cross-attention ----
        # Q: [B,T,D], K/V: [B,N,D]
        attn_out, attn_weights = self.cross_attn(Q, K, V)       # attn_out: [B,T,D]

        # ---- Step 3: broadcast to N tokens ----
        attn_expanded = attn_out.unsqueeze(1).expand(B, N, self.T, D)   # [B,N,T,D]
        scene_expanded = scene_feat.unsqueeze(2).expand(B, N, self.T, D)  # [B,N,T,D]

        # ---- Step 4: MLP 得到 mask logits ----
        feat = torch.cat([scene_expanded, attn_expanded], dim=-1)  # [B,N,T,2D]
        logits = self.mlp(feat).squeeze(-1)                        # [B,N,T]

        mask = torch.sigmoid(logits)  # (0,1)

        # ---- Step 5: mask embedding ----
        mask_emb = self.mask_emb_proj(mask.unsqueeze(-1)) * self.emb_scale
        # → [B,N,T,1] → [B,N,T,D_model]

        return mask, mask_emb


if __name__ == "__main__":
    # 简单测试一下
    B = 2
    N = 5
    D = 32
    T_fut = 10
    d_model = 4

    scene_feat = torch.randn(B, N, D)

    model = MaskNet(
        hidden_dim=D,
        T_fut=T_fut,
        d_model=d_model,
    )

    mask, mask_emb = model(scene_feat)

    print("mask shape:", mask)          # 期望: [B, N, T]
    print("mask_emb shape:", mask_emb.shape)  # 期望: [B, N, T, d_model]