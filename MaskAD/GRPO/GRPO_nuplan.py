import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.distributions import Normal

from MaskAD.model.maskplanner import MaskPlanner

# === 新增：从 utils 中引入通用函数 ===
from MaskAD.GRPO.utils import (
    load_config_from_yaml,
    inspect_gradients,
    cosine_beta_schedule,
    extract,
    make_timesteps,
    build_physical_states_from_future,
    compute_grpo_trajectory_reward,
    compute_total_grad_norm,
)


# =========================
# GRPO + BC Loss 微调版
# =========================
class MaskPlannerGRPO(MaskPlanner):
    """
    基于 MaskPlanner 的第二阶段：用 GRPO + BC loss 微调 decoder (DiT)
    只更新 decoder，encoder 与 mask_net 冻结。
    """
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters(config)
        self._future_len = config.future_len
        self._agents_len = 1 + config.predicted_neighbor_num
        self._diffusion_steps = config.diffusion_steps

        # ========= GRPO 超参数 =========
        self.group_size = getattr(config, "group_size", 6)
        self.gamma_denoising = getattr(config, "gamma_denoising", 0.6)
        self.clip_advantage_lower_quantile = getattr(config, "clip_adv_lower_q", 0.0)
        self.clip_advantage_upper_quantile = getattr(config, "clip_adv_upper_q", 1.0)
        self.min_sampling_denoising_std = getattr(config, "min_sampling_std", 0.04)
        self.min_logprob_denoising_std = getattr(config, "min_logprob_std", 0.1)
        self.randn_clip_value = getattr(config, "randn_clip_value", 5.0)
        self.final_state_clip_value = getattr(config, "final_state_clip_value", 1.0)
        self.beta_kl = getattr(config, "beta_kl", 0.0)

        self.grpo_num_inference_steps = getattr(
            config, "grpo_num_inference_steps", self._diffusion_steps
        )

        # ========= BC 超参数 =========
        self.bc_coeff = getattr(config, "bc_coeff", 0.1)
        self.use_bc_loss = getattr(config, "use_bc_loss", True)

        # ========= 旧策略 decoder (teacher) =========
        self.old_decoder = copy.deepcopy(self.decoder)
        self.old_decoder.eval()
        for p in self.old_decoder.parameters():
            p.requires_grad = False

        # ========= 冻结 Encoder & MaskNet，只微调 decoder =========
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.mask_net.parameters():
            p.requires_grad = False

        print(">>> Encoder frozen. GRPO will NOT update encoder.")
        print(">>> MaskNet frozen. GRPO will NOT update MaskNet.")
        print(">>> Decoder remains trainable (GRPO will update Decoder).")

        # ========= 初始化 DDPM 系数 =========
        self._init_ddpm_for_grpo(self._diffusion_steps)

    def on_train_epoch_end(self):
        """
        每个 epoch 结束后，把当前 decoder 的参数拷贝给 old_decoder。
        如果你想 teacher 固定为第一阶段 BC 模型，可以注释掉这一步。
        """
        self.old_decoder.load_state_dict(copy.deepcopy(self.decoder.state_dict()))
        self.old_decoder.eval()
        for p in self.old_decoder.parameters():
            p.requires_grad = False

    # ----------------- DDPM schedule & 系数 -----------------
    def _init_ddpm_for_grpo(self, num_train_timesteps: int):
        betas = cosine_beta_schedule(num_train_timesteps)
        self.register_buffer("ddpm_betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("ddpm_alphas", alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("ddpm_alphas_cumprod", alphas_cumprod)

        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=alphas_cumprod.device), alphas_cumprod[:-1]]
        )
        self.register_buffer("ddpm_alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("ddpm_sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "ddpm_sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

        var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("ddpm_var", var)
        self.register_buffer("ddpm_logvar_clipped", torch.log(var.clamp(min=1e-20)))

        mu_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        mu_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (
            1.0 - alphas_cumprod
        )
        self.register_buffer("ddpm_mu_coef1", mu_coef1)
        self.register_buffer("ddpm_mu_coef2", mu_coef2)

    # ----------------- p(x_{t-1}|x_t) 单步 -----------------
    def p_mean_variance_step(
        self,
        x_t_future: torch.Tensor,          # [B',P,T,4] (normalized future)
        t: torch.Tensor,                   # [B']
        encoder_outputs_rep: dict,
        batch_rep: dict,
        current_states_rep: torch.Tensor,  # [B',P,4]
        decoder: nn.Module = None,
    ):
        if decoder is None:
            decoder = self.decoder

        Bp, P, T, D = x_t_future.shape

        x_t_full = torch.cat(
            [current_states_rep[:, :, None, :], x_t_future], dim=2
        )  # [B',P,1+T,4]

        # 这里的 diffusion_time 与你一阶段训练中的定义保持一致
        diffusion_time = (t.float() + 1.0) / float(self._diffusion_steps)   # [B']
        merged_inputs = {
            **batch_rep,
            "sampled_trajectories": x_t_full,
            "diffusion_time": diffusion_time,
        }

        decoder_outputs = decoder(encoder_outputs_rep, merged_inputs)
        x0_full = decoder_outputs["x_start"]           # [B',P,1+T,4]
        x0_future = x0_full[:, :, 1:, :].clamp(-1.0, 1.0)

        mu_coef1 = extract(self.ddpm_mu_coef1, t, x_t_future.shape)
        mu_coef2 = extract(self.ddpm_mu_coef2, t, x_t_future.shape)

        mean = mu_coef1 * x0_future + mu_coef2 * x_t_future
        logvar = extract(self.ddpm_logvar_clipped, t, x_t_future.shape)

        return mean, logvar, x0_future

    # ----------------- 采样反向链 x_T → x_0 -----------------
    @torch.no_grad()
    def sample_chain_grpo(
        self,
        encoder_outputs_rep: dict,
        batch_rep: dict,
        current_states_rep: torch.Tensor,
        deterministic: bool = False,
        decoder: nn.Module = None,
    ):
        if decoder is None:
            decoder = self.decoder

        Bp = current_states_rep.shape[0]
        P = current_states_rep.shape[1]
        T = self._future_len
        device = current_states_rep.device

        current_future = torch.randn(
            Bp, P, T, 4, device=device, dtype=current_states_rep.dtype
        )
        chains = [current_future.clone()]

        step_size = max(self._diffusion_steps // self.grpo_num_inference_steps, 1)
        timesteps = list(reversed(range(0, self._diffusion_steps, step_size)))
        num_steps = len(timesteps)

        for i, t_int in enumerate(timesteps):
            t_batch = make_timesteps(Bp, t_int, device)

            mean, logvar, _ = self.p_mean_variance_step(
                current_future,
                t_batch,
                encoder_outputs_rep,
                batch_rep,
                current_states_rep,
                decoder=decoder,
            )

            std = torch.exp(0.5 * logvar)
            if deterministic or t_int == 0:
                std = torch.zeros_like(std)
            else:
                std = std.clamp(min=self.min_sampling_denoising_std)

            noise = torch.randn_like(current_future)
            if self.randn_clip_value is not None:
                noise = noise.clamp(-self.randn_clip_value, self.randn_clip_value)

            current_future = mean + std * noise

            if i == num_steps - 1 and self.final_state_clip_value is not None:
                current_future = current_future.clamp(
                    -self.final_state_clip_value, self.final_state_clip_value
                )

            chains.append(current_future.clone())

        chains = torch.stack(chains, dim=1)  # [B*G,K+1,P,T,4]
        final_future_norm = current_future   # [B*G,P,T,4]

        return chains, final_future_norm

    # ----------------- 整条链的 log p(x_{t-1}|x_t) -----------------
    def get_logprobs_chain(
        self,
        encoder_outputs_rep: dict,
        batch_rep: dict,
        current_states_rep: torch.Tensor,
        chains: torch.Tensor,
        decoder: nn.Module = None,
    ) -> torch.Tensor:
        if decoder is None:
            decoder = self.decoder

        Bp, K1, P, T, D = chains.shape
        device = chains.device
        num_steps = K1 - 1

        step_size = max(self._diffusion_steps // self.grpo_num_inference_steps, 1)
        timesteps = list(reversed(range(0, self._diffusion_steps, step_size)))[:num_steps]

        logprob_list = []

        for s, t_int in enumerate(timesteps):
            x_t = chains[:, s]
            x_tm1 = chains[:, s + 1]
            t_batch = make_timesteps(Bp, t_int, device)

            mean, logvar, _ = self.p_mean_variance_step(
                x_t,
                t_batch,
                encoder_outputs_rep,
                batch_rep,
                current_states_rep,
                decoder=decoder,
            )

            std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
            dist = Normal(mean, std)
            log_prob_s = dist.log_prob(x_tm1)   # [B*G,P,T,4]
            logprob_list.append(log_prob_s)

        log_probs = torch.stack(logprob_list, dim=1)   # [B*G,K,P,T,4]
        log_probs_flat = log_probs.view(-1, P, T, D)   # [B*G*K,P,T,4]
        return log_probs_flat

    # ----------------- GRPO 主逻辑 + BC Loss -----------------
    def forward_grpo_diffusion(self, batch: dict, deterministic: bool = False):
        encoder_outputs = self.encoder(batch)
        B = batch["ego_current_state"].shape[0]
        Pn = self.cfg.predicted_neighbor_num
        P = 1 + Pn
        G = self.group_size
        device = batch["ego_current_state"].device

        ego_current = batch["ego_current_state"][:, None, :4]          # [B,1,4]
        neighbors_current = batch["agents_past"][:, 1:Pn+1, -1, :4]    # [B,Pn,4]
        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B,P,4]

        # mask 网络（GRPO 中保留，但 encoder & mask_net 已冻结）
        mask, mask_emb = self.mask_net(
            encoder_outputs["encoding"],
            encoder_outputs["encoding_agents"],
        )
        encoder_outputs["mask"] = mask
        encoder_outputs["mask_emb"] = mask_emb

        # B → B*G
        def repeat_B_to_BG(x):
            if isinstance(x, torch.Tensor):
                x_rep = x.unsqueeze(1).repeat(1, G, *([1] * (x.ndim - 1)))  # [B,G,...]
                return x_rep.view(B * G, *x.shape[1:])
            return x

        encoder_outputs_rep = {k: repeat_B_to_BG(v) for k, v in encoder_outputs.items()}
        batch_rep = {k: repeat_B_to_BG(v) for k, v in batch.items()}
        current_states_rep = repeat_B_to_BG(current_states)   # [B*G,P,4]

        # 采样链（当前策略）
        chains, final_future_norm_BG = self.sample_chain_grpo(
            encoder_outputs_rep,
            batch_rep,
            current_states_rep,
            deterministic=deterministic,
            decoder=self.decoder,
        )  # chains: [B*G,K+1,P,T,4]

        # 构造物理轨迹 [x,y,θ,vx,vy]
        final_states5_BG = build_physical_states_from_future(
            final_future_norm_BG,
            self.state_normalizer,
        )  # [B*G,P,T,5]

        final_states5_GB = (
            final_states5_BG.view(B, G, P, self._future_len, 5)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )  # [G,B,P,T,5]

        # Reward & Advantage
        rewards_GB = compute_grpo_trajectory_reward(
            final_states5_GB,
            batch=batch,
            v_target=5.0,
            collision_dist=2.0,
        )  # [G,B]

        rewards = rewards_GB.permute(1, 0).contiguous()  # [B,G]

        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards - mean_r) / std_r).view(-1).detach()  # [B*G]

        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(adv_min, adv_max)  # [B*G]

        num_denoising_steps = chains.shape[1] - 1
        step_idx = torch.arange(num_denoising_steps, device=device)
        discount = self.gamma_denoising ** (num_denoising_steps - step_idx - 1)

        adv_steps = (
            advantages.view(B, G, 1).expand(B, G, num_denoising_steps)
        )  # [B,G,K]
        discount = (
            discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)
        )
        adv_weighted_flat = (adv_steps * discount).reshape(-1)  # [B*G*K]

        # 当前策略 logprob 链
        log_probs_flat = self.get_logprobs_chain(
            encoder_outputs_rep,
            batch_rep,
            current_states_rep,
            chains,
            decoder=self.decoder,
        )  # [B*G*K,P,T,4]
        log_probs = log_probs_flat.clamp(min=-5, max=2).mean(dim=[1, 2, 3])  # [B*G*K]

        policy_loss = -(log_probs * adv_weighted_flat).mean()
        total_loss = policy_loss

        bc_loss = torch.tensor(0.0, device=device)

        # BC Loss：teacher = old_decoder
        if self.use_bc_loss:
            with torch.no_grad():
                teacher_chains, _ = self.sample_chain_grpo(
                    encoder_outputs_rep,
                    batch_rep,
                    current_states_rep,
                    deterministic=False,
                    decoder=self.old_decoder,
                )  # [B*G,K+1,P,T,4]

            bc_logp_flat = self.get_logprobs_chain(
                encoder_outputs_rep,
                batch_rep,
                current_states_rep,
                teacher_chains,
                decoder=self.decoder,
            )  # [B*G*K,P,T,4]

            K_steps = teacher_chains.shape[1] - 1
            bc_logp = bc_logp_flat.clamp(min=-5, max=2)
            bc_logp = bc_logp.view(
                -1, K_steps, bc_logp.shape[1], bc_logp.shape[2], bc_logp.shape[3]
            )
            bc_logp = bc_logp.mean(dim=[1, 2, 3, 4])  # [B*G]

            bc_loss = -bc_logp.mean()
            total_loss = total_loss + self.bc_coeff * bc_loss

        log_dict = {
            "policy_loss": policy_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "reward_mean": rewards.mean().detach(),
        }
        log_dict["loss"] = total_loss.detach()

        return total_loss, log_dict

    # 重载 Lightning 的训练 / 验证，直接走 GRPO
    def training_step(self, batch, batch_idx):
        loss, log_dict = self.forward_grpo_diffusion(batch, deterministic=False)
        train_log = {f"train/{k}": v for k, v in log_dict.items()}
        self.log_dict(train_log, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_grpo_diffusion(batch, deterministic=True)
        val_log = {f"val/{k}": v for k, v in log_dict.items()}
        self.log_dict(val_log, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss


# =========================
# main: 简单 GRPO + BC 测试
# =========================
def test_grpo():
    """
    简单的 GRPO + BC Loss 测试：
    - 构造一个 fake batch
    - 前向跑一次 forward_grpo_diffusion
    - backward，看梯度是否正常
    """
    cfg_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/nuplan.yaml"
    config = load_config_from_yaml(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化 GRPO 模型
    model = MaskPlannerGRPO(config).to(device)

    batch_size = 2

    batch = {
        "ego_current_state": torch.randn(batch_size, 10).float(),
        "agents_past": torch.randn(batch_size, 33, 21, 11).float(),

        "static_objects": torch.randn(batch_size, 5, 10).float(),

        "lanes": torch.randn(batch_size, 70, 20, 12).float(),
        "lanes_speed_limit": torch.randn(batch_size, 70, 1).float(),
        "lanes_has_speed_limit": torch.randint(0, 2, (batch_size, 70, 1)).bool(),

        "route_lanes": torch.randn(batch_size, 25, 20, 12).float(),
        "route_lanes_speed_limit": torch.randn(batch_size, 25, 1).float(),
        "route_lanes_has_speed_limit": torch.randint(0, 2, (batch_size, 25, 1)).bool(),

        "neighbor_agents_future": torch.randn(batch_size, 32, config.future_len, 3).float(),
        "ego_agent_future": torch.randn(batch_size, config.future_len, 3).float(),
    }

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 前向（no grad）看数值
    model.eval()
    with torch.no_grad():
        loss, log_dict = model.forward_grpo_diffusion(batch, deterministic=False)

    print("=== GRPO 前向检查（no grad） ===")
    print("total loss:", loss.item())
    for k, v in log_dict.items():
        print(f"{k}: {float(v)}")

    # 测试反向传播
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss, log_dict = model.forward_grpo_diffusion(batch, deterministic=False)
    loss.backward()

    total_grad_norm = compute_total_grad_norm(model)

    print("\n=== GRPO 反向传播检查 ===")
    print("total loss:", float(loss.detach().cpu()))
    print("total grad norm:", total_grad_norm)
    for k, v in log_dict.items():
        print(f"{k}: {float(v.detach().cpu())}")

    # 检查一下 mask_net 是否有梯度（应该都是 False）
    print("\n=== MaskNet 梯度检查（应为冻结） ===")
    for name, p in model.named_parameters():
        if "mask_net" in name:
            has_grad = p.grad is not None
            grad_norm = p.grad.norm().item() if has_grad else None
            print(f"{name:60s} | has_grad={has_grad} | grad_norm={grad_norm}")

    # 打印所有参数梯度信息（可选）
    inspect_gradients(model)

    optimizer.step()

    print("\n=== GRPO + BC 测试结束 ===")


if __name__ == "__main__":
    test_grpo()
