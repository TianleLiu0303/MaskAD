import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """
    Transformer-style block with cross-attention only:
      x = x + CrossAttn(LN(x), context)
      x = x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)

        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, context, context_key_padding_mask=None):
        """
        x:       [B, Q, D]
        context: [B, N_ctx, D]
        """
        h = self.ln1(x)
        attn_out, _ = self.cross_attn(
            query=h,
            key=context,
            value=context,
            key_padding_mask=context_key_padding_mask,  # [B, N_ctx], True = PAD (optional)
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class MaskNet(nn.Module):
    """
    Inputs:
        context_feat: [B, N_ctx, D]  scene tokens
        agent_feat:   [B, N, D]      agent tokens (for producing N queries)
    Output:
        mask_flat:    [B, N*T]       flattened mask (values in [0,1])
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        T_fut: int = 80,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.D = hidden_dim
        self.T = T_fut

        # (1) Learnable time query bank: [T, D]
        # One learnable query vector per future timestep.
        self.time_queries = nn.Parameter(torch.randn(T_fut, hidden_dim) * 0.02)

        # (2) Agent-conditioned modulation: make queries agent-specific
        # agent_feat -> per-agent offset added to each time query
        self.agent_to_q = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # (3) Cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # (4) Head: scalar logit per query token
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        context_feat: torch.Tensor,
        agent_feat: torch.Tensor,
        context_mask: torch.Tensor = None,
    ):
        """
        context_feat: [B, N_ctx, D]
        agent_feat:   [B, N, D]
        context_mask: [B, N_ctx] True for padding tokens (optional)
        """
        B, N_ctx, D = context_feat.shape
        B2, N, D2 = agent_feat.shape
        assert B == B2 and D == self.D and D2 == self.D

        # ---------------------------------------------------------
        # Build learnable queries for each (agent, time)
        # ---------------------------------------------------------
        # base time queries: [T, D] -> [1, 1, T, D] -> [B, N, T, D]
        q_time = self.time_queries.view(1, 1, self.T, self.D).expand(B, N, self.T, self.D)

        # agent modulation: [B, N, D] -> [B, N, 1, D] -> [B, N, T, D]
        q_agent = self.agent_to_q(agent_feat).unsqueeze(2).expand(B, N, self.T, self.D)

        # final query tokens: [B, N, T, D] -> flatten -> [B, N*T, D]
        q = (q_time + q_agent).reshape(B, N * self.T, self.D)

        # ---------------------------------------------------------
        # Cross-attention: queries attend to scene context
        # ---------------------------------------------------------
        x = q
        for blk in self.blocks:
            x = blk(x, context_feat, context_key_padding_mask=context_mask)

        # ---------------------------------------------------------
        # Head -> logits -> sigmoid -> flatten mask [B, N*T]
        # ---------------------------------------------------------
        logits = self.head(x).squeeze(-1)     # [B, N*T]
        mask_flat = torch.sigmoid(logits)     # [B, N*T] in [0,1]

        return mask_flat


if __name__ == "__main__":
    B = 2
    N_ctx = 336
    N = 64
    D = 256
    T = 80

    context_feat = torch.randn(B, N_ctx, D)
    agent_feat = torch.randn(B, N, D)

    model = MaskNet(
        hidden_dim=D,
        T_fut=T,
        num_layers=4,
        num_heads=8,
        ffn_ratio=4.0,
        dropout=0.0,
        mlp_hidden=256,
    )

    mask_flat = model(context_feat, agent_feat)
    print("mask_flat shape:", mask_flat.shape)  # [B, N*T]
