import copy
import torch
import lightning.pytorch as pl
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss, cross_entropy

from DPR.model.Encoder import Encoder
from DPR.model.Decoder import Decoder, GoalPredictor
from DPR.model.utils import DDPM_Sampler
from DPR.model.model_utils import (
    inverse_kinematics,
    roll_out,
    batch_transform_trajs_to_global_frame,
)


class DPRDiffusionGRPO(pl.LightningModule):
    """
    Diffusion + GRPO 版本的 DPR：
    - 仍然使用 Encoder / Decoder / GoalPredictor / DDPM_Sampler
    - GRPO 采用 diffusion 链 + logprob 的形式（类似 ReCogDrive）
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # --------- 基本配置 ---------
        self._future_len = cfg["future_len"]
        self._agents_len = cfg["agents_len"]
        self._action_len = cfg["action_len"]
        self._diffusion_steps = cfg["diffusion_steps"]
        self._encoder_layers = cfg["encoder_layers"]
        self._encoder_version = cfg.get("encoder_version", "v1")
        self._action_mean = cfg["action_mean"]
        self._action_std = cfg["action_std"]

        self._train_encoder = cfg.get("train_encoder", True)
        self._train_denoiser = cfg.get("train_denoiser", True)
        self._train_predictor = cfg.get("train_predictor", True)
        self._with_predictor = cfg.get("with_predictor", False)
        self._prediction_type = cfg.get("prediction_type", "sample")
        self._schedule_type = cfg.get("schedule_type", "cosine")
        self._replay_buffer = cfg.get("replay_buffer", False)
        encoder_path = cfg.get("encoder_ckpt", None)

        # --------- 编码器 ---------
        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
        if encoder_path is not None:
            model_dict = torch.load(encoder_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            for key in list(model_dict.keys()):
                if not key.startswith("encoder."):
                    del model_dict[key]
            print("Load Encoder Weights from:", encoder_path)
            self.encoder.load_state_dict(model_dict, strict=False)

        # --------- 解码器（denoiser） ---------
        self.denoiser = Decoder(
            decoder_drop_path_rate=cfg["decoder_drop_path_rate"],
            action_len=cfg["action_len"],
            predicted_agents_num=cfg["predicted_agents_num"],
            future_len=self._future_len,
            hidden_dim=cfg["hidden_dim"],
            decoder_depth=cfg["decoder_depth"],
            num_heads=cfg["num_heads"],
        )

        # 旧策略（可用于未来的 BC / KL）
        self.old_denoiser = copy.deepcopy(self.denoiser)
        self.old_denoiser.eval()
        for p in self.old_denoiser.parameters():
            p.requires_grad = False

        # --------- 行为先验预测器 ---------
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        # --------- 噪声调度器（DDPM） ---------
        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
        )

        # --------- GRPO 超参数 ---------
        self.group_size = cfg.get("group_size", 6)  # 每个样本生成 G 条轨迹
        self.gamma_denoising = cfg.get("gamma_denoising", 0.6)
        self.clip_advantage_lower_quantile = cfg.get("clip_adv_lower_q", 0.0)
        self.clip_advantage_upper_quantile = cfg.get("clip_adv_upper_q", 1.0)
        self.min_sampling_denoising_std = cfg.get("min_sampling_std", 0.04)
        self.min_logprob_denoising_std = cfg.get("min_logprob_std", 0.1)
        self.randn_clip_value = cfg.get("randn_clip_value", 5.0)
        self.final_action_clip_value = cfg.get("final_action_clip_value", 1.0)
        self.beta_kl = cfg.get("beta", 0.0)  # 如果以后想加 KL 惩罚，可以用

        # 采样链的步数（可以小于训练步数）
        self.grpo_num_inference_steps = cfg.get(
            "grpo_num_inference_steps", self._diffusion_steps
        )

        # --------- 归一化参数 ---------
        self.register_buffer("action_mean", torch.tensor(self._action_mean))
        self.register_buffer("action_std", torch.tensor(self._action_std))

        # --------- GRPO 用的 DDPM buffer ---------
        self._init_ddpm_for_grpo(self._diffusion_steps)

    # ======================================================
    # 优化器
    # ======================================================
    def configure_optimizers(self):
        if not self._train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if not self._train_denoiser:
            for p in self.denoiser.parameters():
                p.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for p in self.predictor.parameters():
                p.requires_grad = False

        params_to_update = [p for p in self.parameters() if p.requires_grad]
        assert len(params_to_update) > 0, "No parameters to update"

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        lr_warmup_step = self.cfg["lr_warmup_step"]
        lr_step_freq = self.cfg["lr_step_freq"]
        lr_step_gamma = self.cfg["lr_step_gamma"]

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma**n

            lr_scale = max(min(lr_scale, 1.0), 1e-2)
            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, lr_warmup_step, lr_step_freq, lr_step_gamma
            ),
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ======================================================
    # DDPM buffer & 工具函数（GRPO链用）
    # ======================================================
    def _init_ddpm_for_grpo(self, num_train_timesteps: int):
        betas = self.cosine_beta_schedule(num_train_timesteps)
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
        self.register_buffer("ddpm_sqrt_one_minus_alphas_cumprod",torch.sqrt(1.0 - alphas_cumprod))

        var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("ddpm_var", var)
        self.register_buffer(
            "ddpm_logvar_clipped", torch.log(var.clamp(min=1e-20))
        )
        mu_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        mu_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("ddpm_mu_coef1", mu_coef1)
        self.register_buffer("ddpm_mu_coef2", mu_coef2)


    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32):
        steps = timesteps + 1
        x = torch.linspace(0, steps, steps, dtype=dtype)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0, 0.999)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        '''a must be shape [T], t must be shape [B]'''
        b, *_ = t.shape  
        out = a.gather(-1, t) # out[i] = a[t[i]]
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def make_timesteps(batch_size: int, t: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), t, device=device, dtype=torch.long)

    # ======================================================
    # Denoiser & Predictor 前向（沿用你原来的写法）
    # ======================================================
    def forward_denoiser(
        self,
        encoder_outputs: dict,
        noised_actions_normalized: torch.Tensor,
        diffusion_step: torch.Tensor,
        agents_future_41: torch.Tensor,
    ):
        """
        这里保持跟你原来 forward_denoiser 的逻辑一致，
        只是把 reference_model 的部分省略（因为 GRPO 走了新的链式逻辑）。
        """
        # 归一化动作 -> 反归一化
        noised_actions = self.unnormalize_actions(noised_actions_normalized)

        denoiser_output = self.denoiser(
            encoder_outputs, noised_actions, diffusion_step, agents_future_41
        )

        # 如果 Decoder 返回 dict，这里需要取真正的 score
        # 例如：denoiser_output = denoiser_output["score"]
        # 下同
        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output,
            diffusion_step,
            noised_actions_normalized,
            prediction_type=self._prediction_type,
        )

        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        assert (
            encoder_outputs["agents"].shape[1] >= self._agents_len
        ), "代理数量过多"

        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)

        denoised_trajs = roll_out(
            current_states,
            denoised_actions,
            action_len=self.denoiser._action_len,
            global_frame=True,
        )

        return {
            "denoiser_output": denoiser_output,
            "denoised_actions_normalized": denoised_actions_normalized,
            "denoised_actions": denoised_actions,
            "denoised_trajs": denoised_trajs,
        }

    def forward_predictor(self, encoder_outputs: dict):
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)

        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        assert (
            encoder_outputs["agents"].shape[1] >= self._agents_len
        ), "代理数量过多"

        goal_actions = self.unnormalize_actions(goal_actions_normalized)

        goal_trajs = roll_out(
            current_states[:, :, None, :],
            goal_actions,
            action_len=self.predictor._action_len,
            global_frame=True,
        )

        return {
            "goal_actions_normalized": goal_actions_normalized,
            "goal_actions": goal_actions,
            "goal_scores": goal_scores,
            "goal_trajs": goal_trajs,
        }

    # ======================================================
    # Diffusion 链：p(x_{t-1}|x_t) + sample_chain + logprobs
    # ======================================================
    def p_mean_variance_step(
        self,
        x_t_normalized: torch.Tensor,  # [B, A, T, 2]
        t: torch.Tensor,  # [B]
        encoder_outputs: dict,
        agents_future_41: torch.Tensor,  # [B, A, 41, 5]
    ):
        """
        单步 p(x_{t-1}|x_t, cond)，使用当前 denoiser + DDPM 公式。
        """
        Bp, A, T, D = x_t_normalized.shape
        device = x_t_normalized.device

        diffusion_steps_full = (
            t.view(Bp, 1, 1, 1).repeat(1, A, 1, 1)
        )  # [B', A, 1, 1]

        # 反归一化
        noised_actions = self.unnormalize_actions(x_t_normalized)

        denoiser_output = self.denoiser(
            encoder_outputs,
            noised_actions,
            diffusion_steps_full.view(Bp, A),
            agents_future_41,
        )

        # 如果是 dict，这里要取真正的 score
        # denoiser_output_score = denoiser_output["score"] 之类
        denoised_x0_normalized = self.noise_scheduler.q_x0(
            denoiser_output,
            diffusion_steps_full.view(Bp, A),
            x_t_normalized,
            prediction_type=self._prediction_type,
        )

        denoised_x0_normalized = denoised_x0_normalized.clamp(
            -1.0, 1.0
        )  # 你原来 denoised_clip_value = 1.0

        mu_coef1 = self.extract(self.ddpm_mu_coef1, t, x_t_normalized.shape)
        mu_coef2 = self.extract(self.ddpm_mu_coef2, t, x_t_normalized.shape)
        mean = mu_coef1 * denoised_x0_normalized + mu_coef2 * x_t_normalized

        logvar = self.extract(self.ddpm_logvar_clipped, t, x_t_normalized.shape)
        return mean, logvar, denoised_x0_normalized

    @torch.no_grad()
    def sample_chain_grpo(
        self,
        encoder_outputs_rep: dict,  # [B*G,...]
        agents_future_41_rep: torch.Tensor,  # [B*G, A, 41, 5]
        deterministic: bool = False,
    ):
        """
        从纯噪声开始，生成一条完整的反向链：
        chains: [B*G, K+1, A, T, 2] (归一化动作)
        final_actions: [B*G, A, T, 2] (物理动作)
        final_trajs: [B*G, A, T_traj, 5] (轨迹)
        """
        Bp = agents_future_41_rep.shape[0]
        A = self._agents_len
        T = self._future_len
        device = agents_future_41_rep.device

        current_actions = torch.randn(
            Bp,
            A,
            T,
            2,
            device=device,
            dtype=self.action_mean.dtype,
        )
        chains = [current_actions.clone()]

        step_size = max(self._diffusion_steps // self.grpo_num_inference_steps, 1)
        timesteps = list(reversed(range(0, self._diffusion_steps, step_size)))
        print(f"GRPO sampling with {len(timesteps)} steps (step size {step_size}).")
        num_steps = len(timesteps)

        for i, t_int in enumerate(timesteps):
            t_batch = self.make_timesteps(Bp, t_int, device)
            mean, logvar, _ = self.p_mean_variance_step(
                current_actions, t_batch, encoder_outputs_rep, agents_future_41_rep
            )

            std = torch.exp(0.5 * logvar)
            if deterministic or t_int == 0:
                std = torch.zeros_like(std)
            else:
                std = std.clamp(min=self.min_sampling_denoising_std)

            noise = torch.randn_like(current_actions)
            if self.randn_clip_value is not None:
                noise = noise.clamp(-self.randn_clip_value, self.randn_clip_value)

            current_actions = mean + std * noise

            if i == num_steps - 1 and self.final_action_clip_value is not None:
                current_actions = current_actions.clamp(
                    -self.final_action_clip_value,
                    self.final_action_clip_value,
                )

            chains.append(current_actions.clone())

        chains = torch.stack(chains, dim=1)  # [B*G, K+1, A, T, 2]

        final_actions = self.unnormalize_actions(current_actions)  # [B*G, A, T, 2]
        current_states = encoder_outputs_rep["agents"][
            :, : self._agents_len, -1
        ]  # [B*G, A, 5]
        final_trajs = roll_out(
            current_states,
            final_actions,
            action_len=self.denoiser._action_len,
            global_frame=True,
        )

        return chains, final_actions, final_trajs

    def get_logprobs_chain(
        self,
        encoder_outputs_rep: dict,
        agents_future_41_rep: torch.Tensor,
        chains: torch.Tensor,
    ) -> torch.Tensor:
        """
        对整条链计算 log p(x_{t-1}|x_t, cond)
        chains: [B*G, K+1, A, T, 2]
        返回: [B*G*K, A, T, 2]
        """
        Bp, K1, A, T, D = chains.shape
        device = chains.device
        num_steps = K1 - 1

        step_size = max(self._diffusion_steps // self.grpo_num_inference_steps, 1)
        timesteps = list(reversed(range(0, self._diffusion_steps, step_size)))[
            :num_steps
        ]

        logprob_list = []

        for s, t_int in enumerate(timesteps):
            x_t = chains[:, s]  # [B*G, A, T, 2]
            x_tm1 = chains[:, s + 1]
            t_batch = self.make_timesteps(Bp, t_int, device)

            mean, logvar, _ = self.p_mean_variance_step(
                x_t,
                t_batch,
                encoder_outputs_rep,
                agents_future_41_rep,
            )

            std = torch.exp(0.5 * logvar).clamp(
                min=self.min_logprob_denoising_std
            )
            dist = Normal(mean, std)
            log_prob_s = dist.log_prob(x_tm1)  # [B*G, A, T, 2]
            logprob_list.append(log_prob_s)

        log_probs = torch.stack(logprob_list, dim=1)  # [B*G, K, A, T, 2]
        log_probs_flat = log_probs.view(-1, A, T, D)  # [B*G*K, A, T, 2]
        return log_probs_flat

    # ======================================================
    # GRPO 主逻辑：采样 G 条轨迹 → reward → advantage → policy loss
    # ======================================================
    def forward_grpo_diffusion(self, batch: dict, deterministic: bool = False):
        """
        用 Diffusion-GRPO 方式训练：
        - 使用链式 DDPM 采样
        - compute_grpo_trajectory_reward 计算 reward
        - 用链上 log prob 做 policy gradient
        """
        # ------------ GT 处理（和你原 forward_and_get_loss 开头一样） ------------
        agents_future = batch["agents_future"][:, : self._agents_len]  # [B, A, 81, 5]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)  # [B, A, 81]
        agents_interested = batch["agents_interested"][:, : self._agents_len]  # [B, A]
        anchors = batch["anchors"][:, : self._agents_len]  # [B, A, 64, 2]

        current = agents_future[:, :, 0:1, :]  # [B, A, 1, 5]
        future_downsampled = agents_future[:, :, 1::2, :]  # [B, A, 40, 5]
        agents_future_41 = torch.cat([current, future_downsampled], dim=2)  # [B, A, 41, 5]

        B = agents_future.shape[0]
        A = self._agents_len
        G = self.group_size
        device = agents_future.device


        # ------------ Encoder 前向 ------------
        encoder_outputs = self.encoder(batch)


        encoder_outputs["agents_type"] =  encoder_outputs["agents_type"].view(B, -1)
        # 将 [B,...] 展开为 [B*G,...]
        def repeat_B_to_BG(x):
            if isinstance(x, torch.Tensor):
                x_rep = x.unsqueeze(1).repeat(1, G, *([1] * (x.ndim - 1)))  # [B, G, ...]
                return x_rep.view(B * G, *x.shape[1:])
            return x

        encoder_outputs_rep = {k: repeat_B_to_BG(v) for k, v in encoder_outputs.items()}
        agents_future_41_rep = repeat_B_to_BG(agents_future_41)

        print(encoder_outputs_rep["encodings"].shape)
        print(agents_future_41_rep.shape)

        # ------------ 采样链 & 轨迹 ------------
        chains, final_actions_BG, final_trajs_BG = self.sample_chain_grpo(
            encoder_outputs_rep,
            agents_future_41_rep,
            deterministic=deterministic,
        )  # chains: [B*G, K+1, A, T, 2]

        # reshape 回 [G, B, ...]，方便用你原来的 reward 函数
        Bp, K1, A, T_action, D_action = chains.shape
        assert Bp == B * G

        final_trajs_GB = (
            final_trajs_BG.view(B, G, A, -1, 5).permute(1, 0, 2, 3, 4)
        )  # [G, B, A, T_traj, 5]
        final_actions_GB = (
            final_actions_BG.view(B, G, A, T_action, D_action).permute(1, 0, 2, 3, 4)
        )  # [G, B, A, T, 2]

        # ------------ Reward & Advantage ------------
        rewards_GB = self.compute_grpo_trajectory_reward(
            final_trajs_GB,
            final_actions_GB,
            v_target=5.0,
            collision_dist=2.0,
        )  # [G, B]
        rewards = rewards_GB.permute(1, 0).contiguous()  # [B, G]

        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards - mean_r) / std_r).view(-1).detach()  # [B*G]

        adv_min = torch.quantile(
            advantages, self.clip_advantage_lower_quantile
        )
        adv_max = torch.quantile(
            advantages, self.clip_advantage_upper_quantile
        )
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

        # ------------ logprob 链 ------------
        log_probs_flat = self.get_logprobs_chain(
            encoder_outputs_rep, agents_future_41_rep, chains
        )  # [B*G*K, A, T, 2]
        log_probs = log_probs_flat.clamp(min=-5, max=2).mean(dim=[1, 2, 3])  # [B*G*K]

        policy_loss = -(log_probs * adv_weighted_flat).mean()
        total_loss = policy_loss

        log_dict = {
            "train/policy_loss": policy_loss.detach(),
            "train/reward_mean": rewards.mean().detach(),
        }

        # ------------ 可选：KL 惩罚（当前策略 vs 旧策略）------------
        if self.beta_kl > 0.0:
            with torch.no_grad():
                chains_old, _, _ = self.sample_chain_grpo(
                    encoder_outputs_rep, agents_future_41_rep, deterministic=True
                )
            # 这里可以实现一个 KL term（略），目前不加
            pass

        # ------------ metrics（沿用你原来的 calculate_metrics_denoise）------------
        denoise_ade, denoise_fde = self.calculate_metrics_denoise(
            final_trajs_GB,
            agents_future,
            agents_future_valid,
            agents_interested,
            top_k=8,
        )
        log_dict.update(
            {
                "train/denoise_ADE": denoise_ade,
                "train/denoise_FDE": denoise_fde,
            }
        )

        # ------------ GoalPredictor 监督训练（和你原来的类似）------------
        if self._train_predictor and self.predictor is not None:
            goal_outputs = self.forward_predictor(encoder_outputs)
            goal_scores = goal_outputs["goal_scores"]
            goal_trajs = goal_outputs["goal_trajs"]

            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs,
                goal_scores,
                agents_future,
                agents_future_valid,
                anchors,
                agents_interested,
            )
            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss = total_loss + 1.0 * pred_loss

            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
                top_k=8,
            )
            log_dict.update(
                {
                    "train/goal_loss": goal_loss_mean.detach(),
                    "train/score_loss": score_loss_mean.detach(),
                    "train/pred_ADE": pred_ade,
                    "train/pred_FDE": pred_fde,
                }
            )

        log_dict["train/loss"] = total_loss.detach()
        return total_loss, log_dict

    # ======================================================
    # Lightning 训练 / 验证步骤
    # ======================================================
    def training_step(self, batch, batch_idx):
        loss, log_dict = self.forward_grpo_diffusion(batch, deterministic=False)
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_grpo_diffusion(batch, deterministic=True)
        val_log_dict = {k.replace("train/", "val/"): v for k, v in log_dict.items()}
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        # 更新 old_denoiser（未来如需 KL / BC 可用）
        self.old_denoiser.load_state_dict(copy.deepcopy(self.denoiser.state_dict()))
        self.old_denoiser.eval()

    # ======================================================
    # 评估 & reward & 工具函数（几乎原样搬过来）
    # ======================================================
    ### denoise的损失
    @torch.no_grad()
    def calculate_metrics_denoise(
        self,
        denoised_trajs,  # [G, B, A, T, 5]
        agents_future,
        agents_future_valid,
        agents_interested,
        top_k=None,
    ):
        if not top_k:
            top_k = self._agents_len

        # 取均值轨迹
        denoised_trajs_avg = denoised_trajs.mean(dim=0)[..., :2]  # [B, A, T, 2]
        pred_traj = denoised_trajs_avg[:, :top_k, :, :]  # [B, A, T, 2]

        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        gt_mask = (
            agents_future_valid[:, :top_k, 1:]
            & (agents_interested[:, :top_k, None] > 0)
        ).bool()

        denoise_mse = torch.norm(pred_traj - gt, dim=-1)  # [B, A, T]

        denoise_ADE = denoise_mse[gt_mask].mean()
        denoise_FDE = denoise_mse[..., -1][gt_mask[..., -1]].mean()

        return denoise_ADE.item(), denoise_FDE.item()

    ### predictor的损失
    def goal_loss(
        self,
        trajs,
        scores,
        agents_future,
        agents_future_valid,
        anchors,
        agents_interested,
    ):
        current_states = agents_future[:, :, 0, :3]
        anchors_global = batch_transform_trajs_to_global_frame(
            anchors, current_states
        )
        num_batch, num_agents, num_query, _ = anchors_global.shape

        traj_mask = agents_future_valid[..., 1:] * (
            agents_interested[..., None] > 0
        )

        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1)
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)
        trajs = trajs.flatten(0, 1)[..., :3]
        anchors_global = anchors_global.flatten(0, 1)

        idx_anchor = torch.argmin(
            torch.norm(anchors_global - goal_gt, dim=-1), dim=-1
        )

        dist = torch.norm(
            trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1
        )
        dist = dist * traj_mask.flatten(0, 1)[:, None, :]
        idx = torch.argmin(dist.mean(-1), dim=-1)

        idx = torch.where(
            agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx
        )
        trajs_select = trajs[torch.arange(num_batch * num_agents), idx]

        traj_loss = smooth_l1_loss(
            trajs_select, trajs_gt, reduction="none"
        ).sum(-1)
        traj_loss = traj_loss * traj_mask.flatten(0, 1)

        scores = scores.flatten(0, 1)
        score_loss = cross_entropy(scores, idx, reduction="none")
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0)

        traj_loss_mean = traj_loss.sum() / traj_mask.sum()
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum()

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_predict(
        self,
        goal_trajs,
        agents_future,
        agents_future_valid,
        agents_interested,
        top_k=None,
    ):
        if not top_k:
            top_k = self._agents_len

        goal_trajs = goal_trajs[:, :top_k, :, :, :2]  # [B, A, Q, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]

        gt_mask = (
            agents_future_valid[:, :top_k, 1:]
            & (agents_interested[:, :top_k, None] > 0)
        ).bool()

        goal_mse = torch.norm(
            goal_trajs - gt[:, :, None, :, :], dim=-1
        )  # [B, A, Q, T]
        goal_mse = goal_mse * gt_mask[..., None, :]

        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)  # [B, A]

        best_goal_mse = goal_mse[
            torch.arange(goal_mse.shape[0])[:, None],
            torch.arange(goal_mse.shape[1])[None, :],
            best_idx,
        ]  # [B, A, T]

        goal_ADE = best_goal_mse.sum() / gt_mask.sum()
        goal_FDE = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum()

        return goal_ADE.item(), goal_FDE.item()

    def compute_grpo_trajectory_reward(
        self,
        trajectories,  # [G, B, N, T, 5]
        actions=None,  # [G, B, N, A, 2]
        v_target=5.0,
        collision_dist=2.0,
        weights=None,
    ):
        """
        你原来的 reward 函数，原样搬过来。
        """
        G, B, N, T, D = trajectories.shape
        assert D == 5, "Each state must be [x, y, theta, v_x, v_y]"

        if weights is None:
            weights = {
                "smoothness": 1.0,
                "speed": 1.0,
                "orientation": 1.0,
                "collision": 2.0,
                "action_accel": 1.0,
                "action_yaw": 1.0,
            }

        x, y, theta, v_x, v_y = (
            trajectories[..., 0],
            trajectories[..., 1],
            trajectories[..., 2],
            trajectories[..., 3],
            trajectories[..., 4],
        )
        v = torch.sqrt(v_x**2 + v_y**2)

        acc = v[..., 1:] - v[..., :-1]
        smoothness_reward = -torch.mean(acc**2, dim=(2, 3))  # [G, B]

        speed_diff = (v - v_target) ** 2
        speed_reward = -torch.mean(speed_diff, dim=(2, 3))  # [G, B]

        d_theta = theta[..., 1:] - theta[..., :-1]
        orientation_reward = -torch.mean(d_theta**2, dim=(2, 3))  # [G, B]

        pos = torch.stack([x, y], dim=-1)  # [G, B, N, T, 2]
        collision_penalty = torch.zeros(G, B, device=trajectories.device)
        for g in range(G):
            for b in range(B):
                min_dist = []
                for t in range(T):
                    dist = torch.cdist(pos[g, b, :, t], pos[g, b, :, t], p=2)
                    mask = ~torch.eye(
                        N, dtype=torch.bool, device=trajectories.device
                    )
                    min_d = dist[mask].min()
                    min_dist.append(min_d)
                min_dist = torch.stack(min_dist)
                penalty = torch.mean(
                    (collision_dist - min_dist).clamp(min=0.0) ** 2
                )
                collision_penalty[g, b] = penalty

        if actions is not None:
            G2, B2, N2, A2, D2 = actions.shape
            assert (G2, B2, N2) == (G, B, N), "Action tensor shape mismatch"
            accel = actions[..., 0]
            yaw_rate = actions[..., 1]
            accel_penalty = -torch.mean(accel**2, dim=(2, 3))
            yaw_penalty = -torch.mean(yaw_rate**2, dim=(2, 3))
        else:
            accel_penalty = torch.zeros(G, B, device=trajectories.device)
            yaw_penalty = torch.zeros(G, B, device=trajectories.device)

        total_reward = (
            weights["smoothness"] * smoothness_reward
            + weights["speed"] * speed_reward
            + weights["orientation"] * orientation_reward
            - weights["collision"] * collision_penalty
            + weights["action_accel"] * accel_penalty
            + weights["action_yaw"] * yaw_penalty
        )

        return total_reward

    # ======================================================
    # 辅助：batch_to_device / normalize / unnormalize
    # ======================================================
    def batch_to_device(self, input_dict: dict, device: torch.device = "cuda"):
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.to(device)
        return input_dict

    def normalize_actions(self, actions: torch.Tensor):
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor):
        return actions * self.action_std + self.action_mean


###############################################################
#  Test p_mean_variance_step() 独立测试脚本
###############################################################
if __name__ == "__main__":
    import torch

    # ====== 构造简化 cfg ======
    cfg = {
        "future_len": 80,
        "agents_len": 32,
        "action_len": 40,
        "diffusion_steps": 32,
        "encoder_layers": 3,
        "action_mean": [0.0, 0.0],
        "action_std": [1.0, 1.0],
        "decoder_drop_path_rate": 0.0,
        "predicted_agents_num": 32,
        "hidden_dim": 256,
        "decoder_depth": 2,
        "num_heads": 4,
        "embeding_dim": 5,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    }

    print(">>> 创建模型...")
    model = DPRDiffusionGRPO(cfg)
    model.eval()

    # ====== 构造 fake 输入数据 ======
    Bp = 4          # 测试 batch (B * G)
    A  = cfg["agents_len"]
    T  = cfg["action_len"]

    print(">>> 构造 fake x_t_normalized...")
    x_t_normalized = torch.randn(Bp, A, T, 2)

    print(">>> 构造 fake t...")
    t = torch.randint(0, cfg["diffusion_steps"], (Bp,))

    # fake encoder outputs (只需要必要字段)
    print(">>> 构造 fake encoder_outputs...")
    encoder_outputs = {
        "agents": torch.randn(Bp, A, 11, 8),      # DPR encoder 的常用 shape
        "anchors": torch.randn(Bp, 32, 64, 2),
        "agents_type": torch.randint(0, 5, (Bp, A)),
        "encodings": torch.randn(Bp, 336, 256),
        "agents_mask": torch.randint(0, 5, (Bp, A)), # 假
        "maps_mask": torch.randn(Bp, 256), # 假
        "traffic_light_mask": torch.rand1n(Bp, 16),  # 假
        "relation_encodings": torch.randn(Bp, 336, 336, 32),  # 假
    }

    # fake future 41
    print(">>> 构造 fake agents_future_41...")
    agents_future_41 = torch.randn(Bp, A, 41, 5)

    # ====== 调用测试函数 ======
    print(">>> 调用 p_mean_variance_step()...")
    with torch.no_grad():
        mean, logvar, x0 = model.p_mean_variance_step(
            x_t_normalized, t, encoder_outputs, agents_future_41
        )

    # ====== 打印结果 ======
    print("\n========== 测试结果 ==========")
    print("mean.shape =", mean.shape)
    print("logvar.shape =", logvar.shape)
    print("x0.shape =", x0.shape)

    print("\n>>> 测试完成！")


############################测试函数##########################
# if __name__ == "__main__":
#     """
#     简单的本地测试：
#     - 构造一个 batch_size = 1 的假数据 batch
#     - 实例化 DPRDiffusionGRPO
#     - 跑一遍 forward_grpo_diffusion 看是否能正常前向 / 反向
#     """

#     # ----------------- 构造一个最小 cfg -----------------
#     cfg = {
#         # 时间和代理设置
#         "future_len": 81,          # agents_future: [B, A, 81, 5]
#         "agents_len": 64,          # 每个场景最多 64 个 agent
#         "action_len": 40,          # 你 inverse_kinematics 之后得到 40 个 action
#         "diffusion_steps": 50,     # DDPM 步数
#         "encoder_layers": 4,
#         "encoder_version": "v1",

#         # 动作归一化参数（这里随便设一个例子，实际用你训练时的）
#         "action_mean": [0.0, 0.0],
#         "action_std": [1.0, 1.0],

#         # 模型结构
#         "decoder_drop_path_rate": 0.0,
#         "predicted_agents_num": 64,
#         "hidden_dim": 256,
#         "decoder_depth": 4,
#         "num_heads": 8,

#         # 是否训练各模块
#         "train_encoder": True,
#         "train_denoiser": True,
#         "train_predictor": False,      # 先关掉 predictor，简化测试
#         "with_predictor": False,

#         # 扩散相关
#         "prediction_type": "sample",
#         "schedule_type": "cosine",
#         "replay_buffer": False,
#         "embeding_dim": 5,
#         "encoder_ckpt": None,         # 如果有预训练 encoder 权重可以填路径

#         # GRPO 超参
#         "group_size": 3,              # 每个场景生成 G 条轨迹，小一点方便测试
#         "gamma_denoising": 0.6,
#         "clip_adv_lower_q": 0.0,
#         "clip_adv_upper_q": 1.0,
#         "min_sampling_std": 0.04,
#         "min_logprob_std": 0.1,
#         "randn_clip_value": 5.0,
#         "final_action_clip_value": 1.0,
#         "beta": 0.0,                  # 先不加 KL 惩罚
#         "grpo_num_inference_steps": 10,

#         # 优化 related
#         "lr": 1e-4,
#         "weight_decay": 1e-4,
#         "lr_warmup_step": 1000,
#         "lr_step_freq": 10000,
#         "lr_step_gamma": 0.9,
#     }

#     # ----------------- 实例化模型 -----------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DPRDiffusionGRPO(cfg).to(device)
#     model.train()  # 测试训练分支

#     # ----------------- 构造一个假的 batch（B=1） -----------------
#     B = 2
#     A = cfg["agents_len"]
#     future_len = cfg["future_len"]

#     # 这些 shape 对应你之前打印的单样本数据：
#     # agents_history: (64, 11, 8)
#     # agents_interested: (64,)
#     # agents_type: (64,)
#     # agents_future: (64, 81, 5)
#     # traffic_light_points: (16, 3)
#     # polylines: (256, 30, 5)
#     # polylines_valid: (256,)
#     # relations: (336, 336, 3)
#     # anchors: (64, 64, 2)  # 每个 agent 一个 [64,2] 的 anchor 集合

#     batch = {
#         "agents_history": torch.randn(B, A, 11, 8, device=device),
#         "agents_interested": torch.ones(B, A, device=device),          # 先全 1 当作都感兴趣
#         "agents_type": torch.randint(0, 4, (B, A), device=device),     # 0~3 随机类型
#         "agents_future": torch.randn(B, A, future_len, 5, device=device),
#         "traffic_light_points": torch.randn(B, 16, 3, device=device),
#         "polylines": torch.randn(B, 256, 30, 5, device=device),
#         "polylines_valid": torch.ones(B, 256, device=device),
#         "relations": torch.randn(B, 336, 336, 3, device=device),
#         "anchors": torch.randn(B, A, 64, 2, device=device),            # [B,64,64,2]
#     }

#     # ----------------- 跑一遍 GRPO 前向 -----------------
#     # 注意：这里只是测试 forward 能否跑通，不做真实训练
#     loss, log_dict = model.forward_grpo_diffusion(batch, deterministic=False)

#     print("\n=== Forward GRPO Diffusion Test ===")
#     print("loss:", float(loss.detach().cpu()))
#     for k, v in log_dict.items():
#         if isinstance(v, torch.Tensor):
#             v = v.detach().cpu().item()
#         print(f"{k}: {v}")
