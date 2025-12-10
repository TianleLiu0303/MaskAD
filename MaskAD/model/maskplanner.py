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
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
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

class MaskPlanner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.cfg = config
        # self.save_hyperparameters(self.cfg)
        self.encoder = Diffusion_Encoder(config)
        self.decoder = Diffusion_Decoder(config)
        # self.sampler = DDPM_Sampler(config)

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
        #         param.requires_grad = False

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
        推理阶段：
        - 用 encoder 提取场景特征
        - 用 MaskNet 生成时间-智能体的 mask / mask_emb
        - 构造带有当前状态的初始噪声轨迹 x_T (形状 [B, P, 1+T, 4])
        - 在未来步上叠加 mask_emb
        - 用 DPM-Solver 采样得到 x_0
        - 反归一化 + 去掉第 0 步，输出未来轨迹
        """

        # ============ 1. 观测归一化 + 编码 ============
        batch = self.obs_normalizer(batch)
        encoder_outputs = self.encoder(batch)

        # 场景编码 (encoding: [B, N_scene, D], encoding_agents: [B, P, D])
        scene_enc = encoder_outputs["encoding"]
        agent_enc = encoder_outputs["encoding_agents"]

        # ============ 2. MaskNet：得到 mask / mask_emb ============
        # mask: [B, P, T], mask_emb: [B, P, T, 4]   (这里 T = self._future_len)
        mask, mask_emb = self.mask_net(scene_enc, agent_enc)

        # ============ 3. 当前状态：ego + neighbor ============
        B = batch["ego_current_state"].shape[0]
        Pn = self.cfg.predicted_neighbor_num   # 邻居数
        T = self.cfg.future_len

        # ego 当前：取前 4 维 (x, y, cos, sin) 或你的约定
        ego_current = batch["ego_current_state"][:, None, :4]                          # [B, 1, 4]
        neighbors_current = batch["agents_past"][:, 1:Pn+1, -1, :4]             # [B, Pn, 4]

        current_states = torch.cat([ego_current, neighbors_current], dim=1)           # [B, P, 4]
        B, P, _ = current_states.shape
        assert P == 1 + Pn, f"current_states P={P}, expected 1+{Pn}"

        # 给 encoder_outputs 里也塞一下 mask（如果 decoder 里要用）
        encoder_outputs["mask"] = mask
        encoder_outputs["mask_emb"] = mask_emb

        # ============ 4. 构造初始 x_T (带当前帧 + 噪声未来帧) ============
        # 噪声在未来 T 步，形状 [B, P, T, 4]
        noise = torch.randn(B, P, T, 4, device=current_states.device) * 0.5

        # 拼接当前帧：x_T_raw: [B, P, 1+T, 4]
        xT = torch.cat(
            [current_states[:, :, None, :], noise],   # [B,P,1,4] + [B,P,T,4]
            dim=2
        )

        # ============ 5. mask_emb 只作用于未来步 ============
        # 先在时间维度前面补一个 0 帧，让 mask_emb_full: [B, P, 1+T, 4]
        zero_mask = torch.zeros(B, P, 1, 4, device=mask_emb.device, dtype=mask_emb.dtype)
        # mask_emb: [B,P,T,4]  →  [B,P,1+T,4]
        mask_emb_full = torch.cat([zero_mask, mask_emb], dim=2)

        # 在未来步上做加法（包括 0 帧，但 0 帧是 0，不影响）
        xT = xT + mask_emb_full    # [B, P, 1+T, 4]

        # ============ 6. reshape 给 DPM-Solver 使用 ============
        # DPM-Solver 一般假设 batch 在第 0 维，后面全部看作 data 维
        # 所以把 B 和 P 合并，时间和通道展平
        xT_flat = xT.view(B, P, (1 + T) * 4)    # [B*P, (1+T)*4]

        # ============ 7. 约束函数：强制第 0 帧等于 current_states ============
        def initial_state_constraint(xt, t, step):
            """
            xt: [B*P, (1+T)*4] 的形状
            把它 reshape 回 [B, P, 1+T, 4]，修正第 0 帧，再展平返回。
            """
            xt = xt.view(B, P, 1 + T, 4)
            xt[:, :, 0, :] = current_states   # 强制当前状态不被扩散
            return xt.view(B, P, -1)

        # 一些 decoder / DiT 需要的条件输入
        ego_neighbor_encoding = encoder_outputs["encoding"]    # [B,P,D]，你原来的命名
        route_lanes = batch["route_lanes"]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0  # [B,Pn]

        # ============ 8. 调用 dpm_sampler 进行采样 ============
        x0_flat = dpm_sampler(
            model=self.decoder.decoder.dit,         # 或者 self.decoder.dit，按你目前代码实际结构来
            x_T=xT_flat,
            other_model_params={
                "cross_c": ego_neighbor_encoding,
                "route_lanes": route_lanes,
                "neighbor_current_mask": neighbor_current_mask,
            },
            dpm_solver_params={
                "correcting_xt_fn": initial_state_constraint,
            },
            model_wrapper_params={
                "classifier_fn": None,
                "classifier_kwargs": {
                    "model": self.decoder.decoder.dit,
                    "model_condition": {
                        "cross_c": ego_neighbor_encoding,
                        "route_lanes": route_lanes,
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

        # ============ 9. 反 reshape + 反归一化 + 去掉第 0 帧 ============
        x0 = x0_flat.view(B, P, 1 + T, 4)           # [B,P,1+T,4]
        x0 = self.state_normalizer.inverse(x0)      # 回到真实物理量空间
        x0 = x0[:, :, 1:, :]                        # 去掉当前帧，只保留未来 [B,P,T,4]

        return {
            "prediction": x0
        }


    def training_step(self, batch, batch_idx):
        """
        模型的训练步骤。
        """        
        loss, log_dict = self.forward_train(batch)
        train_log = {f"train_{k}": v for k, v in log_dict.items()}
        self.log_dict(train_log, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.
        """
        loss, log_dict = self.forward_train(batch)
        val_log = {f"val_{k}": v for k, v in log_dict.items()}
        self.log_dict(val_log, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def forward_train(self, batch):

        encoder_outputs = self.encoder(batch)
        
        ego_future = batch['ego_agent_future']                # [B, T, D]
        neighbors_future = batch['neighbor_agents_future']    # [B, P-1, T, D]

        batch, ego_future, neighbors_future = self.aug(batch, ego_future, neighbors_future)

        batch = self.obs_normalizer(batch)

        ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
        neighbor_future_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
        # neighbors_future[neighbor_future_mask] = 0.

        gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future], dim=1) # [B, P, T, D]
        gt_future = self.state_normalizer(gt_future)



       #  mask: torch.Size([2, 33, 80]), mask_emb: torch.Size([2, 33, 80, 4])
        mask, mask_emb = self.mask_net(encoder_outputs['encoding'], encoder_outputs['encoding_agents'])
        encoder_outputs['mask'] = mask
        encoder_outputs['mask_emb'] = mask_emb

        B, Pn, T, _ = neighbors_future.shape
        ego_current, neighbors_current = batch["ego_current_state"][:, :4], batch["agents_past"][:, 1:Pn+1, -1, :4]
        print("ego_current, neighbor_current", ego_current.shape, neighbors_current.shape)

        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

        all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)
        all_gt[:, 1:][neighbor_mask] = 0.0

        eps = 1e-3
        t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]

        mean, std = self.sde.marginal_prob(all_gt[..., 1:, :], t)
        std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

       
        
        z = torch.randn_like(gt_future, device=gt_future.device)  # [B, A, T, D]

        xT = mean + std * z
        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

        merged_inputs = {
            **batch,
            "sampled_trajectories": xT,
            "diffusion_time": t,
        }
        decoder_outputs = self.decoder(encoder_outputs, merged_inputs)

        pred = decoder_outputs['x_start']  # [B, P, (future_len + 1) * 4]

        # 拼成 [B, A, T, D]，保证与 pred 对齐


        # ======== Split Ego & Neighbors ========
        pred_ego = pred[:, 0, 1:, :]             # [B, T, D]
        gt_ego   = gt_future[:, 0, :, :]

        pred_neighbors = pred[:, 1:, 1:, :]      # [B, A-1, T, D]
        gt_neighbors   = gt_future[:, 1:, :, :]

        # ======== Loss Calculation ========
  # ======== 4. 从 mask 中取出 ego / neighbor 对应的部分 ========
        mask_ego = mask[:, 0, :]        # [B, T]
        mask_nbr = mask[:, 1:, :]       # [B, A-1, T]

        # 我们用 (1 - mask) 作为 imitation 的权重：越确定 → 权重越大
        weight_ego = 1.0 - mask_ego                 # [B, T]
        weight_nbr = 1.0 - mask_nbr                 # [B, A-1, T]

        # 为了和误差维度对齐，最后一维加一个维度
        weight_ego = weight_ego.unsqueeze(-1)       # [B, T, 1]
        weight_nbr = weight_nbr.unsqueeze(-1)       # [B, A-1, T, 1]

        # 防止全 0 掉梯度
        eps = 1e-6

        # ======== 5. 加权 ego loss ========
        ego_loss_raw = smooth_l1_loss(
            pred_ego, gt_ego, reduction='none'      # [B, T, D]
        )
        # 广播 weight_ego: [B,T,1] → [B,T,D]
        ego_loss_weighted = ego_loss_raw * weight_ego
        ego_loss = ego_loss_weighted.sum() / (weight_ego.sum() * ego_loss_raw.shape[-1] + eps)

        # ======== 6. 加权 neighbor loss ========
        neighbor_loss_raw = smooth_l1_loss(
            pred_neighbors, gt_neighbors, reduction='none'   # [B, A-1, T, D]
        )
        # weight_nbr: [B,A-1,T,1] → 广播到 [B,A-1,T,D]
        neighbor_loss_weighted = neighbor_loss_raw * weight_nbr
        neighbor_loss = neighbor_loss_weighted.sum() / (weight_nbr.sum() * neighbor_loss_raw.shape[-1] + eps)

        # ======== 7. 汇总 ========
        total_loss = self.cfg.ego_neighbor_weight * ego_loss + neighbor_loss

        log_dict = {
            "loss": total_loss,
            "ego_loss": ego_loss.detach(),
            "neighbor_loss": neighbor_loss.detach(),
        }

        return total_loss, log_dict
 
 
######################test##############################

from omegaconf import OmegaConf

def load_config_from_yaml(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return cfg

def inspect_gradients(model):
    """
    打印每个参数的:
      - 完整名字
      - 是否需要梯度 (requires_grad)
      - 是否得到了梯度 (grad is None?)
      - 梯度范数 (grad_norm)
    """
    print("\n========== 模型梯度检查 ==========\n")
    for name, p in model.named_parameters():
        req = p.requires_grad
        has_grad = p.grad is not None
        grad_norm = p.grad.norm().item() if has_grad else None
        print(f"{name:60s} | req_grad={req} | has_grad={has_grad} | grad_norm={grad_norm}")
    print("\n========== 结束 ==========\n")



def test():
    cfg_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/nuplan.yaml"
    config = load_config_from_yaml(cfg_path)

    # ====== 1. 实例化模型 ======
    model = MaskPlanner(config).cuda()

    batch_size = 2          # batch size

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

        "neighbor_agents_future": torch.randn(batch_size, 32, 80, 3).float(),
        "ego_agent_future": torch.randn(batch_size, 80, 3).float(),
    }
    
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    # ====== 2. 先 eval 一个前向 + loss 看数值 ======
    model.eval()
    with torch.no_grad():
        loss, log_dict = model.forward_train(batch)

    print("=== 前向检查 ===")
    print("total loss (no grad):", loss.item())
    for k, v in log_dict.items():
        print(f"{k}: {v.item()}")

    # ====== 3. 测试梯度能不能正常反向传播 ======
    model.train()  # 切回训练模式

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss, log_dict = model.forward_train(batch)  # 带图的 forward
    loss.backward()

    # 3.1 整体 grad 范数
    total_grad_sq = 0.0
    for _, p in model.named_parameters():
        if p.grad is not None:
            total_grad_sq += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_sq ** 0.5
    print("\n=== 反向传播检查 ===")
    print("total grad norm:", total_grad_norm)

    # 3.2 重点看看 mask_net 有没有梯度
    print("\n=== MaskNet 梯度检查 ===")
    for name, p in model.named_parameters():
        if "mask_net" in name:
            has_grad = p.grad is not None
            grad_norm = p.grad.norm().item() if has_grad else None
            print(f"{name:60s} | has_grad={has_grad} | grad_norm={grad_norm}")

    # 3.3 打印所有模块、所有参数的梯度状态
    inspect_gradients(model)

    # 3.4 简单走一步优化（确认不会报错）
    optimizer.step()

    # ====== 4. 测试 forward 输出形状 ======
    model.eval()
    batch_v2 = {
        "ego_current_state": torch.randn(batch_size, 10).float(),
        "agents_past": torch.randn(batch_size, 33, 21, 11).float(),
    
        "static_objects": torch.randn(batch_size, 5, 10).float(),

        "lanes": torch.randn(batch_size, 70, 20, 12).float(),
        "lanes_speed_limit": torch.randn(batch_size, 70, 1).float(),
        "lanes_has_speed_limit": torch.randint(0, 2, (batch_size, 70, 1)).bool(),

        "route_lanes": torch.randn(batch_size, 25, 20, 12).float(),
        "route_lanes_speed_limit": torch.randn(batch_size, 25, 1).float(),
        "route_lanes_has_speed_limit": torch.randint(0, 2, (batch_size, 25, 1)).bool(),
    }
    for k,v in batch_v2.items():
        if isinstance(v, torch.Tensor):
            batch_v2[k] = v.cuda()
    decoder_outputs = model.forward_inference(batch_v2)   # 你的 forward 目前返回 decoder_outputs dict
    pred = decoder_outputs['prediction']        # [B, P, T, D] or [B, P, (future_len+1)*4]
    print("\n=== forward 输出检查 ===")
    print("pred shape:", pred.shape)




if __name__ == "__main__":
    test()
