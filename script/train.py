import os
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import yaml
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# ====== 模型 & 数据集 ======
from MaskAD.model.maskplanner import MaskPlanner
from MaskAD.GRPO.GRPO_nuplan import MaskPlannerGRPO
from MaskAD.data_process.dataset import MaskADataset


# ================== 1. 数据路径（你可以按需改） ==================
TRAIN_DATA_DIR = "/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/processed_data"
TRAIN_LIST_JSON = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/diffusion_planner.json"

VAL_DATA_DIR = TRAIN_DATA_DIR
VAL_LIST_JSON = TRAIN_LIST_JSON
# 例如未来你可以改成：
# VAL_LIST_JSON = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/diffusion_planner_val.json"


# ================== 2. 加载 yaml ==================
def load_config_from_yaml(cfg_path: Union[str, Path]) -> SimpleNamespace:
    """
    Load a config YAML file into a SimpleNamespace for attribute-style access.
    """
    cfg_path = Path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return SimpleNamespace(**data)


# ================== 3. DataLoader 构建 ==================
def build_dataloaders(cfg: SimpleNamespace):
    """
    用 yaml 中的超参 + 上面写死的路径，构建 train / val DataLoader
    """

    batch_size = cfg.batch_size
    num_workers = getattr(cfg, "num_workers", 4)  # yaml 里没写就给个默认值

    # 过去邻居数量：agent_num 通常 = 1(ego)+32(neighbor)
    past_neighbor_num = cfg.agent_num - 1
    predicted_neighbor_num = cfg.predicted_neighbor_num
    future_len = cfg.future_len

    train_dataset = MaskADataset(
        data_dir=TRAIN_DATA_DIR,
        data_list=TRAIN_LIST_JSON,
        past_neighbor_num=past_neighbor_num,
        predicted_neighbor_num=predicted_neighbor_num,
        future_len=future_len,
    )

    val_dataset = MaskADataset(
        data_dir=VAL_DATA_DIR,
        data_list=VAL_LIST_JSON,
        past_neighbor_num=past_neighbor_num,
        predicted_neighbor_num=predicted_neighbor_num,
        future_len=future_len,
    )

    print(f"[Data] train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ================== 4. 从 IL ckpt 加载 encoder/decoder 权重 ==================
def init_grpo_from_il_ckpt(
    grpo_model: MaskPlannerGRPO,
    il_ckpt_path: Union[str, Path],
):
    """
    从 IL 阶段的 checkpoint 中加载 encoder 和 decoder 的权重到 GRPO 模型中。
    - 只拷贝 'encoder.' 和 'decoder.' 开头的参数
    - 其余（如 optimizer 状态）不加载
    """
    il_ckpt_path = Path(il_ckpt_path)
    assert il_ckpt_path.exists(), f"IL checkpoint not found: {il_ckpt_path}"

    print(f"[GRPO Init] Loading IL checkpoint from: {il_ckpt_path}")
    ckpt = torch.load(il_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    encoder_dict = {}
    decoder_dict = {}

    for k, v in state_dict.items():
        if k.startswith("encoder."):
            # 去掉前缀 "encoder."
            new_k = k[len("encoder."):]
            encoder_dict[new_k] = v
        elif k.startswith("decoder."):
            new_k = k[len("decoder."):]
            decoder_dict[new_k] = v

    missing, unexpected = grpo_model.encoder.load_state_dict(encoder_dict, strict=False)
    print(f"[GRPO Init] Encoder loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    missing, unexpected = grpo_model.decoder.load_state_dict(decoder_dict, strict=False)
    print(f"[GRPO Init] Decoder loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    # old_decoder 会在 __init__ 和 on_train_epoch_end 中由 decoder 拷贝，不需要额外处理


# ================== 5. 主训练入口 ==================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default="/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/nuplan.yaml",
        help="Path to yaml config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="il",
        choices=["il", "grpo"],
        help="Training mode: 'il' for imitation learning, 'grpo' for RL fine-tuning.",
    )
    parser.add_argument(
        "--il_ckpt",
        type=str,
        default=None,
        help="(Optional) IL checkpoint path to init GRPO model's encoder & decoder.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs to use (override cfg.device if given).",
    )

    args = parser.parse_args()

    pl.seed_everything(42)

    # ===== 1. 读取配置 =====
    cfg = load_config_from_yaml(args.cfg)

    # mode 优先级：命令行参数 > cfg.train_mode (如果有) > 默认 "il"
    if args.mode is not None:
        train_mode = args.mode
    else:
        train_mode = getattr(cfg, "train_mode", "il")

    assert train_mode in ["il", "grpo"], f"Unknown train_mode: {train_mode}"
    print(f"[Train] mode = {train_mode}")

    # ===== 2. 实例化 LightningModule =====
    if train_mode == "il":
        model = MaskPlanner(cfg)
        exp_suffix = "IL"
    else:
        model = MaskPlannerGRPO(cfg)

        # 如果给了 IL checkpoint，就用它初始化 encoder & decoder
        if args.il_ckpt is not None:
            init_grpo_from_il_ckpt(model, args.il_ckpt)
        else:
            print("[GRPO Init] No IL ckpt provided, GRPO starts from current weights.")

        exp_suffix = "GRPO"

    # ===== 3. DataLoader =====
    train_loader, val_loader = build_dataloaders(cfg)

    # ===== 4. Logger & Checkpoint =====
    save_dir = cfg.save_dir
    exp_name = cfg.exp_name + f"_{exp_suffix}"

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=exp_name,
    )

    ckpt_dir = Path(save_dir) / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_top_k=3,
        monitor="val/loss",          # 对应 validation_step 里 log 的 key
        mode="min",
        filename="epoch{epoch:02d}-valloss{val/loss:.4f}",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ===== 5. Trainer =====
    max_epochs = getattr(cfg, "epoch", 50)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.devices if args.devices is not None else getattr(cfg, "device", 1)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=getattr(cfg, "grad_clip", 1.0),
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    # ===== 6. 开始训练 =====
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
