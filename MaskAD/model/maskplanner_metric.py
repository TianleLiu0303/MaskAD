import torch
import torch.nn as nn
import lightning.pytorch as pl
from pathlib import Path
from types import SimpleNamespace
import yaml
from MaskAD.model.encoder import Encoder
from MaskAD.model.decoder import Decoder
from MaskAD.model.modules.mask import MaskNet
from MaskAD.utils.data_augmentation import StatePerturbation
from MaskAD.utils.normalizer import ObservationNormalizer, StateNormalizer  
from MaskAD.model.utils import DDPM_Sampler
from MaskAD.model.diffusion_utils.sde import VPSDE_linear
from MaskAD.model.diffusion_utils.sampling import dpm_sampler
from torch.nn.functional import smooth_l1_loss, cross_entropy


### 引入模块metric
from MaskAD.metrics.wosac_submission import WOSACSubmission
from MaskAD.metrics.wosac_metrics import WOSACMetrics
from MaskAD.utils.wosac_utils import get_scenario_rollouts, get_scenario_id_int_tensor

class Diffusion_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.agent_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
    #     self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs

class MaskPlannerMetric(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.cfg = config

        # self.save_hyperparameters(self.cfg)
        self.encoder = Diffusion_Encoder(config)
        self.decoder = Diffusion_Decoder(config)
        # self.sampler = DDPM_Sampler(config)
        self.val_closed_loop = getattr(config, "val_closed_loop", True)
        self.n_rollout_closed_val = getattr(config, "n_rollout_closed_val", 6)
        self.n_batch_wosac_metric = getattr(config, "n_batch_wosac_metric", 999999)

        self.wosac_metrics = WOSACMetrics(prefix="val_closed")   # 只算指标，不存 submission


        self.mask_net = MaskNet(
            hidden_dim=config.hidden_dim,   # scene_feat 的 D 维
            T_fut=config.future_len,        # 未来步数 T
            d_model=4,      # 先用 hidden_dim 做 DiT 的 mask emb 维度
            time_emb_dim=64,
            num_heads=config.num_heads,
            mlp_hidden=128,
        )

        self.aug = StatePerturbation(device="cuda")
        # self.aug = StatePerturbation(device="cpu")
        self.state_normalizer = StateNormalizer.from_json(config)
        self.obs_normalizer = ObservationNormalizer.from_json(config)

        self.sde = VPSDE_linear()

        self.noise_scheduler = DDPM_Sampler(
            steps=self.cfg.diffusion_steps,
            schedule=self.cfg.schedule_type,
            s = self.cfg.schedule_s,
            e = self.cfg.schedule_e,
            tau = self.cfg.schedule_tau,
            scale = self.cfg.schedule_scale,
        )
        #------------------- Optimizer and Scheduler -------------------
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        # if not self._train_encoder:
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False
        # if not self._train_denoiser: 
        #     for param in self.denoiser.parameters():
        #         param.requires_grad = False
        # if self._with_predictor and (not self._train_predictor):
        #     for param in self.predictor.parameters():
        #         param.requires_grad = FalLaneFusionEncoderse

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)              
        
        assert len(params_to_update) > 0, 'No parameters to update'
        
        optimizer = torch.optim.AdamW(
            params_to_update, 
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        
        lr_warmpup_step = self.cfg.lr_warmup_step
        lr_step_freq = self.cfg.lr_step_freq
        lr_step_gamma = self.cfg.lr_step_gamma

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n
        
            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1
        
            return lr_scale
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, 
                lr_warmpup_step, 
                lr_step_freq,
                lr_step_gamma,
            )
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward_inference(self, batch):
        """
        Waymo inference:
        inputs keys:
            - agents_history:        [B, 64, 11, 8]
            - agents_type:           [B, 64]
            - traffic_light_points:  [B, 16, 3]
            - polylines:             [B, 256, 30, 5]
            - route_lanes:           [B, 6, 30, 5]
        输出:
        - prediction: [B, P, T, 4]  (x,y,cos,sin), 其中 P = 1 + predicted_neighbor_num, ego 在 0
        """

        # --------- 0) 一些尺寸 ---------
        B = batch["agents_history"].shape[0]
        Pn = self.cfg.predicted_neighbor_num
        P = 1 + Pn
        T = self.cfg.future_len  # 80

        # --------- 1) obs normalize + encoder ---------
        batch = self.obs_normalizer(batch)
        encoder_outputs = self.encoder(batch)

        # --------- 2) 取前 P 个 agent（ego=0固定）---------
        agents_hist_all = batch["agents_history"]              # [B,64,11,8]
        agents_hist = agents_hist_all[:, :P]                   # [B,P,11,8]

        # --------- 3) current_states: 从 history 最后一帧取 (x,y,yaw) -> (x,y,cos,sin) ---------
        cur_xy = agents_hist[:, :, -1, 0:2]                    # [B,P,2]
        cur_yaw = agents_hist[:, :, -1, 2]                     # [B,P]
        cur_xycs = torch.cat(
            [cur_xy, torch.stack([cur_yaw.cos(), cur_yaw.sin()], dim=-1)],
            dim=-1
        )                                                      # [B,P,4]

        neighbors_current = cur_xycs[:, 1:]                    # [B,P-1,4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :2], 0), dim=-1) == 0  # [B,P-1]

        # --------- 4) MaskNet：对齐到 P 个 agent（很重要）---------
        # 注意：你的 encoder_outputs["encoding_agents"] 很可能是 [B, 64, D]，MaskNet 会输出 [B, 64, T] / [B,64,T,4]
        mask, mask_emb = self.mask_net(encoder_outputs["encoding"], encoder_outputs["encoding_agents"])

        # 只取前 P 个（ego + 近邻）
        mask = mask[:, :P, :]                                  # [B,P,T]
        mask_emb = mask_emb[:, :P, :, :]                       # [B,P,T,4]

        encoder_outputs["mask"] = mask
        encoder_outputs["mask_emb"] = mask_emb

        # --------- 5) 构造初始 x_T：current + noise_future ---------
        # 这里 future 只放 T 步（80步），总长度 1+T
        noise = torch.randn(B, P, T, 4, device=cur_xycs.device) * 0.5
        xT = torch.cat([cur_xycs[:, :, None, :], noise], dim=2)      # [B,P,1+T,4]

        # flatten 成 DiT 接口需要的形状：[B,P,(1+T)*4]
        xT_flat = xT.reshape(B, P, (1 + T) * 4)

        # --------- 6) constraint：强制第0帧等于 current_states ---------
        def initial_state_constraint(xt, t, step):
            xt = xt.view(B, P, 1 + T, 4)
            xt[:, :, 0, :] = cur_xycs
            return xt.view(B, P, -1)

        # --------- 7) DPM-Solver 采样得到 x0 ---------
        # 注意：你的 Decoder.forward 里会把 mask_emb 加到 traj_future，所以这里只传 encoder_outputs 即可
        x0_flat = dpm_sampler(
            model=self.decoder.decoder.dit,      # 你的结构：MaskPlanner -> Diffusion_Decoder -> Decoder -> dit
            x_T=xT_flat,
            other_model_params={
                "cross_c": encoder_outputs["encoding"],          # [B, token_num, D]
                "route_lanes": batch["route_lanes"],             # [B,6,30,5]
                "neighbor_current_mask": neighbor_current_mask,  # [B,P-1]
            },
            dpm_solver_params={
                "correcting_xt_fn": initial_state_constraint,
            },
            model_wrapper_params={
                "classifier_fn": None,
                "classifier_kwargs": {
                    "model": self.decoder.decoder.dit,
                    "model_condition": {
                        "cross_c": encoder_outputs["encoding"],
                        "route_lanes": batch["route_lanes"],
                        "neighbor_current_mask": neighbor_current_mask,
                    },
                    "inputs": batch,
                    "observation_normalizer": self.obs_normalizer,
                    "state_normalizer": self.state_normalizer,
                },
                "guidance_scale": 0.0,
                "guidance_type": "uncond",
            },
        )

        # --------- 8) reshape + 反归一化 + 去掉 current 帧 ---------
        x0 = x0_flat.view(B, P, 1 + T, 4)            # [B,P,1+T,4]
        x0 = self.state_normalizer.inverse(x0)       # 回到真实空间
        pred_future = x0[:, :, 1:, :]                # [B,P,T,4]

        return {
            "prediction": pred_future,               # ego在0号
        }

    def forward_train(self, batch):

        encoder_outputs = self.encoder(batch)

        # --------- 2) 取 GT future：ego + neighbors ---------
        # 选取用于扩散预测的 agent 数：P = 1 + predicted_neighbor_num
        Pn = self.cfg.predicted_neighbor_num
        P = 1 + Pn

        agents_future_all = batch["agents_future"]          # [B,64,81,5]
        agents_hist_all   = batch["agents_history"]         # [B,64,11,8]

        # 只取前 P 个（数据预处理一般已把近邻排在前面；ego=0固定）
        agents_future = agents_future_all[:, :P]            # [B,P,81,5]
        agents_hist   = agents_hist_all[:, :P]              # [B,P,11,8]

        # ego / neighbor future 分开，喂给 aug（保持你原 aug 接口）
        ego_future = agents_future[:, 0, :, :3]             # [B,81,3]  (x,y,yaw)
        neighbors_future = agents_future[:, 1:, :, :3]      # [B,P-1,81,3]

        # --------- 3) 数据增强 + obs_normalizer（保持你原接口）---------
        # batch, ego_future, neighbors_future = self.aug(batch, ego_future, neighbors_future)
        batch = self.obs_normalizer(batch)

        # --------- 4) 把 future 变成 (x,y,cos,sin) 作为扩散 GT ---------
        # neighbor_future_mask: padding 的 timestep（全 0）
        neighbor_future_mask = torch.sum(torch.ne(neighbors_future[...,:-1, :3], 0), dim=-1) == 0  # [B,P-1,81]

        ego_future_xycs = torch.cat(
            [
                ego_future[...,:-1, :2],
                torch.stack([ego_future[...,:-1, 2].cos(), ego_future[...,:-1, 2].sin()], dim=-1),
            ],
            dim=-1,
        )  # [B,81,4]

        neighbors_future_xycs = torch.cat(
            [
                neighbors_future[...,:-1, :2],
                torch.stack([neighbors_future[...,:-1, 2].cos(), neighbors_future[...,:-1, 2].sin()], dim=-1),
            ],
            dim=-1,
        )  # [B,P-1,81,4]

        gt_future = torch.cat([ego_future_xycs[:, None], neighbors_future_xycs], dim=1)  # [B,P,81,4]
        gt_future = self.state_normalizer(gt_future)


        # --------- 5) MaskNet：mask & mask_emb ---------
        mask, mask_emb = self.mask_net(encoder_outputs["encoding"], encoder_outputs["encoding_agents"])
        encoder_outputs["mask"] = mask              # [B,P,81] （确保你的 encoder_agents 对齐 P）
        encoder_outputs["mask_emb"] = mask_emb      # [B,P,81,4]
           # --------- 6) current_states 从 agents_history 的最后一帧取 ---------
        # agents_history: (x,y,yaw,vx,vy,l,w,h) -> current 用 (x,y,yaw) -> 转成 (x,y,cos,sin)
        cur_xy = agents_hist[:, :, -1, 0:2]                      # [B,P,2]
        cur_yaw = agents_hist[:, :, -1, 2]                       # [B,P]
        cur_xycs = torch.cat([cur_xy, torch.stack([cur_yaw.cos(), cur_yaw.sin()], dim=-1)], dim=-1)  # [B,P,4]

        # neighbor_current_mask: 当前帧为 padding 的邻居
        neighbors_current = cur_xycs[:, 1:]                      # [B,P-1,4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :2], 0), dim=-1) == 0  # [B,P-1]
        # 合并：neighbor 的 (current + future) mask，用来把无效 neighbor 置 0
        neighbor_mask = torch.cat([neighbor_current_mask.unsqueeze(-1), neighbor_future_mask], dim=-1)  # [B,P-1,1+81]

        # --------- 7) all_gt = [current + future]，并把无效 neighbor 清零 ---------
        all_gt = torch.cat([cur_xycs[:, :, None, :], gt_future], dim=2)   # [B,P,1+81,4]
        all_gt[:, 1:][neighbor_mask] = 0.0

        # --------- 8) 采样扩散时间 t，构造 noisy x_t（只扩散 future）---------
        B = all_gt.shape[0]
        T = gt_future.shape[2]  # 81
        eps = 1e-3
        t = torch.rand(B, device=all_gt.device) * (1 - eps) + eps          # [B]

        mean, std = self.sde.marginal_prob(all_gt[..., 1:, :], t)          # mean/std: [B,P,T,4] broadcast
        std = std.view(-1, *([1] * (mean.ndim - 1)))

        z = torch.randn_like(gt_future)                                     # [B,P,T,4]
        xT = mean + std * z                                                 # [B,P,T,4]
        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)                    # [B,P,1+T,4]

        # --------- 9) 调 decoder（route_lanes 来自 batch）---------
        merged_inputs = {
            **batch,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "route_lanes": batch["route_lanes"],  # [B,6,30,5]
            # 给 decoder 用的 neighbor mask（如果你 decoder 里用）
            "neighbor_current_mask": neighbor_current_mask,  # [B,P-1]
        }
        decoder_outputs = self.decoder(encoder_outputs, merged_inputs)

        pred = decoder_outputs["x_start"]   # 你 decoder 返回的是 [B,P,1+T,4]

        # --------- 10) ego / neighbors loss（mask 加权）---------
        pred_ego = pred[:, 0, 1:, :]          # [B,T,4]
        gt_ego   = gt_future[:, 0, :, :]      # [B,T,4]

        pred_neighbors = pred[:, 1:, 1:, :]   # [B,P-1,T,4]
        gt_neighbors   = gt_future[:, 1:, :, :]

        # mask: [B,P,T]
        mask_ego = mask[:, 0, :]              # [B,T]
        mask_nbr = mask[:, 1:, :]             # [B,P-1,T]

        weight_ego = (1.0 - mask_ego).unsqueeze(-1)   # [B,T,1]
        weight_nbr = (1.0 - mask_nbr).unsqueeze(-1)   # [B,P-1,T,1]

        # 可选：把 padding neighbor 的 loss 权重直接置 0（更严谨）
        # neighbor_future_mask 是 [B,P-1,T]，True 表示 padding
        weight_nbr = weight_nbr * (~neighbor_future_mask).unsqueeze(-1).float()

        eps_loss = 1e-6
        ego_loss_raw = smooth_l1_loss(pred_ego, gt_ego, reduction="none")          # [B,T,4]
        ego_loss = (ego_loss_raw * weight_ego).sum() / (weight_ego.sum() * ego_loss_raw.shape[-1] + eps_loss)

        nbr_loss_raw = smooth_l1_loss(pred_neighbors, gt_neighbors, reduction="none")  # [B,P-1,T,4]
        neighbor_loss = (nbr_loss_raw * weight_nbr).sum() / (weight_nbr.sum() * nbr_loss_raw.shape[-1] + eps_loss)

        total_loss = self.cfg.ego_neighbor_weight * ego_loss + neighbor_loss

        log_dict = {
            "loss": total_loss,
            "ego_loss": ego_loss.detach(),
            "neighbor_loss": neighbor_loss.detach(),
        }
        return total_loss, log_dict
 
    @torch.no_grad()
    def forward_inference_rollouts(self, batch, n_rollout: int=32):
        """
        返回：
        pred_traj: [B, P, R, T, 2]
        pred_head: [B, P, R, T]
        pred_z:    [B, P, R, T]  (WOSAC 需要 center_z，每步一个；没有就先用 0)
        """
        trajs, heads = [], []
        for _ in range(n_rollout):
            out = self.forward_inference(batch)               # [B,P,T,4]
            xycs = out["prediction"]
            trajs.append(xycs[..., :2])                       # [B,P,T,2]
            heads.append(torch.atan2(xycs[..., 3], xycs[..., 2]))  # [B,P,T]

        pred_traj = torch.stack(trajs, dim=2)   # [B,P,R,T,2]
        pred_head = torch.stack(heads, dim=2)   # [B,P,R,T]

        z0 = batch["agents_z_future"][:, :, 0, 0]          # [B, A]
        pred_z = z0.unsqueeze(-1).expand(-1, -1, 80)       # [B, A, 80]
        pred_z = pred_z.unsqueeze(2).expand(-1, -1, n_rollout, -1)  # [B, A, n_rollout, 80]
    # [B,P,R,T]
        return pred_traj, pred_z, pred_head


    def training_step(self, batch, batch_idx):
        """
        模型的训练步骤。
        """        
        loss, log_dict = self.forward_train(batch)
        train_log = {f"train_{k}": v for k, v in log_dict.items()}
        self.log_dict(train_log, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if not self.val_closed_loop:
            return

        # 1) rollouts
        pred_traj, pred_z, pred_head = self.forward_inference_rollouts(batch)  # [B,P,R,T,2], [B,P,R,T], [B,P,R,T]

        B = pred_traj.shape[0]
        P = pred_traj.shape[1]
        R = pred_traj.shape[2]
        T = pred_traj.shape[3]
        device = pred_traj.device

        # 2) flatten 到 n_agent 形式（WOSAC utils 需要）
        pred_traj = pred_traj.reshape(B * P, R, T, 2)
        pred_z    = pred_z.reshape(B * P, R, T)
        pred_head = pred_head.reshape(B * P, R, T)

        # 3) agent_id / agent_batch / scenario_id
        # 你必须确保 dataloader 提供：
        #   - batch["agents_id"] 或 batch["agent_id"] 类似字段（每个 scenario 的每个 agent 的 object_id）
        #   - batch["tfrecord_path"]：每个 scenario 的 tfrecord 文件路径
        #   - batch["scenario_id"]：每个 scenario 的字符串 id
        agent_id = batch["agents_object_id"][:, :P].reshape(B * P)  # [B*P]
        agent_batch = torch.arange(B, device=device).unsqueeze(1).repeat(1, P).reshape(B * P)  # [B*P]

        scenario_id = get_scenario_id_int_tensor(batch["scenario_id"], device=device)  # [B,16]

        scenario_rollouts = get_scenario_rollouts(
            scenario_id=scenario_id,
            agent_id=agent_id,
            agent_batch=agent_batch,
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
        )

        print("have been here")
        # 4) update metric（注意：需要 tfrecord_path 是 List[str]，长度 B）
        if batch_idx < self.n_batch_wosac_metric:
            self.wosac_metrics.update(batch["tfrecord_path"], scenario_rollouts)
    def on_validation_epoch_end(self):
        metrics = self.wosac_metrics.compute()
        if self.global_rank == 0:
            for k, v in metrics.items():
                self.log(k, v, sync_dist=True, prog_bar=True)  # 或者 self.logger.log_metrics(metrics)
        self.wosac_metrics.reset()



######################test##############################

from omegaconf import OmegaConf

def load_config_from_yaml(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return cfg

def test():
    cfg_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/waymo.yaml"
    config = load_config_from_yaml(cfg_path)

    # ====== 1. 实例化模型 ======
    model = MaskPlannerMetric(config).cuda()

    B = 1  # batch size
    device = "cuda"

    # ====== 2. 构造 Waymo 风格 batch ======
    batch = {
        # ---------- Agents ----------
        # (x, y, yaw, vx, vy, length, width, height)
        "agents_history": torch.randn(B, 64, 11, 8, device=device),

        "agents_z_history": torch.randn(B, 64, 11, 1, device=device),
        
        "agents_z_future": torch.randn(B, 64, 81, 1, device=device),

        # (x, y, yaw, vx, vy)
        "agents_future": torch.randn(B, 64, 81, 5, device=device),

        "agents_type": torch.randint(0, 6, (B, 64), device=device),

        "agents_slot": torch.tensor([
            [  3,   0,   1,   2,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
            16,  17,  18,  19,  20,  21,  22,  23,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]
        ], dtype=torch.int64),

        "agents_object_id": torch.tensor([
            [248, 233, 238, 239, 215, 216, 213, 214, 212, 220, 219, 217, 218, 228, 227, 222,
            211, 224, 225, 223, 232, 221, 226, 230, 231,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1]
        ], dtype=torch.int64),

        # ---------- Static ----------
        # traffic light points (x, y, state)
        "traffic_light_points": torch.randn(B, 16, 3, device=device),

        # ---------- Map ----------
        # polylines: (x, y, heading, traffic_light_state, lane_type)
        "polylines": torch.randn(B, 256, 30, 5, device=device),

        # route lanes: same format as polylines
        "route_lanes": torch.randn(B, 6, 30, 5, device=device),

        # ---------- Meta (可选，但推荐有) ----------
        "scenario_id": ["befb50d537f4b734"]*B,
        "tfrecord_path": ["/mnt/pai-pdc-nas/tianle_DPR/waymo/data_waymo/testing_module/validation_tfexample.tfrecord-00000-of-00150"]*B,
    }

    # ====== 3. eval：前向 + loss（不反传） ======
    model.eval()
    with torch.no_grad():
        model.validation_step(batch, 0)
    model.on_validation_epoch_end()

if __name__ == "__main__":
    test()  