import torch
import torch.nn as nn
import lightning.pytorch as pl
from MaskAD.model.encoder import Encoder
from MaskAD.model.decoder import Decoder
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

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)
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
        decoder_outputs = self.decoder(encoder_outputs, batch)

        pred = decoder_outputs['x_start']  # [B, P, (future_len + 1) * 4]
        
        ego_future = batch['ego_agent_future']                # [B, T, D]
        neighbor_futures = batch['neighbor_agents_future']    # [B, A-1, T, D]

        # 拼成 [B, A, T, D]，保证与 pred 对齐
        target = torch.cat([
            ego_future[:, None, :, :],    # [B, 1, T, D]
            neighbor_futures              # [B, A-1, T, D]
        ], dim=1)

        # ======== Split Ego & Neighbors ========
        pred_ego = pred[:, 0, :, :]             # [B, T, D]
        gt_ego   = target[:, 0, :, :]

        pred_neighbors = pred[:, 1:, :, :]      # [B, A-1, T, D]
        gt_neighbors   = target[:, 1:, :, :]

        # ======== Loss Calculation ========
        ego_loss = smooth_l1_loss(pred_ego, gt_ego, reduction='mean')

        # neighbor 数量可以不固定，只要维度能对齐即可
        neighbor_loss = smooth_l1_loss(pred_neighbors, gt_neighbors, reduction='mean')

        total_loss = self.cfg.ego_neighbor_weight * ego_loss + neighbor_loss

        log_dict = {
            "loss": total_loss,
            "ego_loss": ego_loss.detach(),
            "neighbor_loss": neighbor_loss.detach(),
        }

        return total_loss, log_dict




######################test##############################

def test():
    from types import SimpleNamespace

    # ===== 1. 配一个简单的 config =====
    config = SimpleNamespace()
    config.ego_neighbor_weight = 2.0  # ego loss 的权重
    config.lr = 1e-4                  # 下面用不到，但可以先占位
    config.weight_decay = 1e-2
    config.lr_warmup_step = 1000            
    config.lr_step_freq = 1000
    config.lr_step_gamma = 0.5
 
 ##### encoder 配置 #####
    config.hidden_dim = 192

    config.agent_num = 32               # neighbor_agents_past 里 P 的大小
    config.static_objects_num = 5
    config.lane_num = 70

    config.past_len = 21               # AgentFusionEncoder: time_len
    config.lane_len = 20               # LaneFusionEncoder: lane_len

    config.encoder_drop_path_rate = 0.1
    config.encoder_depth = 2
    config.encoder_tokens_mlp_dim = 64
    config.encoder_channels_mlp_dim = 128

    config.encoder_dim_head = 48
    config.encoder_num_heads = 4
    config.enable_encoder_attn_dist = True
    config.encoder_attn_dropout = 0.1
    

#######  decoder 配置 #####
    config.decoder_drop_path_rate = 0.1
    config.encoder_drop_path_rate = 0.1
    config.hidden_dim = 192
    config.num_heads = 4

    config.predicted_neighbor_num = 32      # P = 1(ego) + 31 = 32
    config.future_len = 80                  # 未来 40 帧 -> (40 + 1) * 4 维度
    config.route_num = 25                 
    config.lane_len = 20                    # 对应 V
    config.decoder_depth = 1  

    
    # ====== 2. 实例化模型，并把 config 挂到 model 上 ======
    model = MaskPlanner(config)

    model.eval()

    batch_size = 2          # batch size

    batch = {
        "ego_current_state": torch.randn(batch_size, 10).float(),

        "ego_agent_future": torch.randn(batch_size, 80, 3).float(),

        "neighbor_agents_past": torch.randn(batch_size, 32, 21, 11).float(),
        "neighbor_agents_future": torch.randn(batch_size, 32, 80, 3).float(),

        "static_objects": torch.randn(batch_size, 5, 10).float(),

        "lanes": torch.randn(batch_size, 70, 20, 12).float(),
        "lanes_speed_limit": torch.randn(batch_size, 70, 1).float(),
        "lanes_has_speed_limit": torch.randint(0, 2, (batch_size, 70, 1)).bool(),

        "route_lanes": torch.randn(batch_size, 25, 20, 12).float(),
        "route_lanes_speed_limit": torch.randn(batch_size, 25, 1).float(),
        "route_lanes_has_speed_limit": torch.randint(0, 2, (batch_size, 25, 1)).bool(),
        "sampled_trajectories": torch.randn(batch_size, 33, 80 * 3).float(),
        "diffusion_time": torch.randint(0, 1000, (batch_size,)).float(),
    }

    # ====== 5. 调用 forward_and_get_loss，打印结果 ======
    loss, log_dict = model.forward_and_get_loss(batch)

    print("total loss:", loss.item())
    for k, v in log_dict.items():
        print(f"{k}: {v.item()}")



if __name__ == "__main__":
    test()