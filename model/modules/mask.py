import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskNet(nn.Module):
    """
    输入:
        context_feat: [B, N_ctx, D]    # 所有场景 token (agents + lanes + static + tl...)
        agent_feat:   [B, N_agent, D]  # 只包含车辆/可控 agent 的 token
    输出:
        mask:      [B, N_agent, T]
        mask_emb:  [B, N_agent, T, d_model]
    """

    def __init__(
        self,
        hidden_dim=192,    # D
        T_fut=80,          # 未来步数 T
        d_model=4,         # 加到 DiT / 轨迹 embedding 的维度
        time_emb_dim=64,
        num_heads=4,
        mlp_hidden=128,
    ):
        super().__init__()

        self.T = T_fut

        # 1. 时间 embedding
        self.time_emb = nn.Embedding(T_fut, time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # 2. cross-attn: time query → context tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # 3. MLP: (agent_feat, time_context) → scalar logit
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

        # 4. mask embedding
        self.mask_emb_proj = nn.Linear(1, d_model)
        self.emb_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, context_feat, agent_feat):
        """
        context_feat: [B, N_ctx, D]
        agent_feat:   [B, N_agent, D]
        """
        B, N_ctx, D = context_feat.shape
        _, N_agent, D_a = agent_feat.shape
        assert D == D_a, "context_feat 和 agent_feat 的通道维度 D 必须一致"

        # K, V 用所有场景 token
        K = context_feat
        V = context_feat

        # ---- Step 1: time embeddings → Q ----
        t_ids = torch.arange(self.T, device=context_feat.device)    # [T]
        t_emb = self.time_emb(t_ids)                               # [T, time_emb_dim]
        t_emb = self.time_proj(t_emb)                              # [T, D]
        Q = t_emb.unsqueeze(0).expand(B, self.T, D)                # [B, T, D]

        # ---- Step 2: cross-attention ----
        # Q: [B,T,D], K/V: [B,N_ctx,D]
        attn_out, attn_weights = self.cross_attn(Q, K, V)          # attn_out: [B,T,D]

        # ---- Step 3: broadcast 到 N_agent ----
        attn_expanded = attn_out.unsqueeze(1).expand(B, N_agent, self.T, D)   # [B,N_agent,T,D]
        agent_expanded = agent_feat.unsqueeze(2).expand(B, N_agent, self.T, D)  # [B,N_agent,T,D]

        # ---- Step 4: MLP → mask logits ----
        feat = torch.cat([agent_expanded, attn_expanded], dim=-1)  # [B,N_agent,T,2D]
        logits = self.mlp(feat).squeeze(-1)                        # [B,N_agent,T]

        mask = torch.sigmoid(logits)                               # [B,N_agent,T]

        # ---- Step 5: mask embedding ----
        mask_emb = self.mask_emb_proj(mask.unsqueeze(-1)) * self.emb_scale
        # [B,N_agent,T,1] → [B,N_agent,T,d_model]

        return mask, mask_emb


if __name__ == "__main__":
    # 简单测试
    B = 2
    N_ctx = 107    # 所有 token 数量
    N_agent = 33   # 车辆数
    D = 192
    T_fut = 80
    d_model = 4

    context_feat = torch.randn(B, N_ctx, D)
    agent_feat   = torch.randn(B, N_agent, D)

    model = MaskNet(
        hidden_dim=D,
        T_fut=T_fut,
        d_model=d_model,
    )

    mask, mask_emb = model(context_feat, agent_feat)

    print("mask shape:", mask.shape)        # [B, N_agent, T]
    print("mask_emb shape:", mask_emb.shape)  # [B, N_agent, T, d_model]
