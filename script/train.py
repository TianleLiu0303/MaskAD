import os
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import yaml
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from MaskAD.model.maskplanner import MaskPlanner
from MaskAD.data_process.dataset import MaskADataset   # 就是你刚才那个 MaskADataset


# ================== 1. 这里填你的数据路径 ==================
# 注意：这几个路径在 yaml 里没有，所以在 train.py 里手动指定

TRAIN_DATA_DIR = "/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/processed_data"
TRAIN_LIST_JSON = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/diffusion_planner.json"

# 如果你现在还没有单独的验证集，可以先暂时用训练集做 val，或者自己再做一个 json
VAL_DATA_DIR = TRAIN_DATA_DIR
VAL_LIST_JSON = TRAIN_LIST_JSON
# 例如你之后可以改成：
# VAL_LIST_JSON = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/diffusion_planner_val.json"


# ================== 2. 加载 yaml ==================
def load_config_from_yaml(cfg_path):
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


# ================== 4. 主训练入口 ==================
def main():
    pl.seed_everything(42)

    # 假设 train.py 放在 MaskAD/ 目录下，yaml 在 MaskAD/config/nuplan.yaml
    cfg_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/nuplan.yaml"

    cfg = load_config_from_yaml(cfg_path)

    # ===== 1. 实例化 LightningModule =====
    model = MaskPlanner(cfg)

    # ===== 2. DataLoader =====
    train_loader, val_loader = build_dataloaders(cfg)

    # ===== 3. Logger & Checkpoint =====
    save_dir = cfg.save_dir
    exp_name = cfg.exp_name

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

    # ===== 4. Trainer =====
    max_epochs = getattr(cfg, "epoch", 50)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = getattr(cfg, "device", 1)

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

    # ===== 5. 开始训练 =====
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
