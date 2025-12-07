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
# from MaskAD.model.utils import DDPM_Sampler
from torch.nn.functional import smooth_l1_loss, cross_entropy


class Diffusion_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
    #     self.initialize_weights()

    # def initialize_weights(self):
    #     # Initialize transformer layers:
    #     def _basic_init(m):
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)
    #         elif isinstance(m, nn.Embedding):
    #             nn.init.normal_(m.weight, mean=0.0, std=0.02)
    #     self.apply(_basic_init)

    #     # Initialize embedding MLP:
    #     nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
    #     nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
    #     nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
    #     nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
    #     self.initialize_weights()

    # def initialize_weights(self):
    #     # Initialize transformer layers:
    #     def _basic_init(m):
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)
    #         elif isinstance(m, nn.Embedding):
    #             nn.init.normal_(m.weight, mean=0.0, std=0.02)
    #     self.apply(_basic_init)

    #     # Initialize timestep embedding MLP:
    #     nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
    #     nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

    #     # Zero-out adaLN modulation layers in DiT blocks:
    #     for block in self.decoder.dit.blocks:
    #         nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    #         nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    #     # Zero-out output layers:
    #     nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
    #     nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
    #     nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
    #     nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs

class MaskPlanner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.cfg = config
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

        self.aug = StatePerturbation()

        self.state_normalizer = StateNormalizer.from_json(config)
        self.obs_normalizer = ObservationNormalizer.from_json(config)

        #------------------- Optimizer and Scheduler -------------------
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_denoiser: 
            for param in self.denoiser.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)              
        
        assert len(params_to_update) > 0, 'No parameters to update'
        
        optimizer = torch.optim.AdamW(
            params_to_update, 
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )
        
        lr_warmpup_step = self.cfg['lr_warmup_step']
        lr_step_freq = self.cfg['lr_step_freq']
        lr_step_gamma = self.cfg['lr_step_gamma']

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

    def forward(self, batch):

        batch = self.obs_normalizer(batch)
        encoder_outputs = self.encoder(batch)
      
        
        mask, mask_emb = self.mask_net(encoder_outputs['encoding'], encoder_outputs['encoding_agents'])
        
        encoder_outputs['mask'] = mask
        encoder_outputs['mask_emb'] = mask_emb

        decoder_outputs = self.decoder(encoder_outputs, batch)

        decoder_outputs['x_start'] = self.state_normalizer.inverse(decoder_outputs['x_start'])

        return decoder_outputs

    def training_step(self, batch, batch_idx):
        """
        模型的训练步骤。
        """        
        # 前向传播并计算损失
        loss, log_dict = self.forward_and_get_loss(batch)
        # 记录日志
        '''on_step=True: 每个训练步骤都会记录日志。
           on_epoch=False: 不在每个 epoch 结束时记录日志。
           sync_dist=True: 在分布式训练中同步日志。
           prog_bar=True: 将日志显示在进度条中，便于实时监控。'''
        # self.log_dict用于批量记录， self.log用于单个记录
        # self.log_dict(log_dict, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        train_log = {f"train/{k}": v for k, v in log_dict.items()}
        self.log_dict(train_log, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.
        """
        loss, log_dict = self.forward_and_get_loss(batch)
        val_log = {f"val/{k}": v for k, v in log_dict.items()}
        self.log_dict(val_log, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        # self.log_dict(log_dict, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        
        return loss

    def forward_and_get_loss(self, batch):

        encoder_outputs = self.encoder(batch)
        
        ego_future = batch['ego_agent_future']                # [B, T, D]
        neighbors_future = batch['neighbor_agents_future']    # [B, A-1, T, D]

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
        mask_neighbor = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
        neighbors_future[mask_neighbor] = 0.

        mask, mask_emb = self.mask_net(encoder_outputs['encoding'], encoder_outputs['encoding_agents'])
        
        #  mask: torch.Size([2, 33, 80]), mask_emb: torch.Size([2, 33, 80, 4])
        encoder_outputs['mask'] = mask
        encoder_outputs['mask_emb'] = mask_emb

        decoder_outputs = self.decoder(encoder_outputs, batch)

        pred = decoder_outputs['x_start']  # [B, P, (future_len + 1) * 4]

        # 拼成 [B, A, T, D]，保证与 pred 对齐
        target = torch.cat([
            ego_future[:, None, :, :],    # [B, 1, T, D]
            neighbors_future              # [B, A-1, T, D]
        ], dim=1)


        target = self.state_normalizer(target)

        # ======== Split Ego & Neighbors ========
        pred_ego = pred[:, 0, :, :]             # [B, T, D]
        gt_ego   = target[:, 0, :, :]

        pred_neighbors = pred[:, 1:, :, :]      # [B, A-1, T, D]
        gt_neighbors   = target[:, 1:, :, :]

        # ======== Loss Calculation ========
  # ======== 4. 从 mask 中取出 ego / neighbor 对应的部分 ========
        # mask: [B, P, T]
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
            "mask_ego_mean": mask_ego.mean().detach(),
            "mask_nbr_mean": mask_nbr.mean().detach(),
        }

        return total_loss, log_dict
 
 
######################test##############################

def load_config_from_yaml(cfg_path):
    """
    Load a config YAML file into a SimpleNamespace for attribute-style access.
    """
    cfg_path = Path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return SimpleNamespace(**data)

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
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "nuplan.yaml"
    config = load_config_from_yaml(cfg_path)

    # ====== 1. 实例化模型 ======
    model = MaskPlanner(config)

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

        "sampled_trajectories": torch.randn(batch_size, 33, 80, 4).float(),
        "diffusion_time": torch.randint(0, 1000, (batch_size,)).float(),
    }

    # ====== 2. 先 eval 一个前向 + loss 看数值 ======
    model.eval()
    with torch.no_grad():
        loss, log_dict = model.forward_and_get_loss(batch)

    print("=== 前向检查 ===")
    print("total loss (no grad):", loss.item())
    for k, v in log_dict.items():
        print(f"{k}: {v.item()}")

    # ====== 3. 测试梯度能不能正常反向传播 ======
    model.train()  # 切回训练模式

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    loss, log_dict = model.forward_and_get_loss(batch)  # 带图的 forward
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
    decoder_outputs = model.forward(batch)   # 你的 forward 目前返回 decoder_outputs dict
    pred = decoder_outputs['x_start']        # [B, P, T, D] or [B, P, (future_len+1)*4]
    print("\n=== forward 输出检查 ===")
    print("pred shape:", pred.shape)




if __name__ == "__main__":
    test()
