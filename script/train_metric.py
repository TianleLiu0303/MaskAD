import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Union, Optional

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf

# ====== 模型 ======
from MaskAD.model.maskplanner_metric import MaskPlannerMetric
from MaskAD.GRPO.GRPO_waymo import MaskPlannerGRPO

# ====== Waymo Dataset ======
# 这里假设你已经把 WaymoDataset / data_collate_fn 放在 MaskAD.data_process.dataset 里
from MaskAD.data_process.dataset import WaymoDataset, data_collate_fn


# ================== config ==================
def load_config_from_yaml(cfg_path: str):
    return OmegaConf.load(cfg_path)


# ================== dataloaders ==================
def build_dataloaders(cfg):
    """
    Waymo dataloader:
      - data_dir 下是一堆 pkl
      - 每个 pkl 里包含你之前打印的字段（agents_history, agents_future, polylines, route_lanes, ...）
    """
    batch_size = int(cfg.batch_size)
    num_workers = int(getattr(cfg, "num_workers", 4))

    train_data_dir = getattr(cfg, "train_data_dir", None)
    val_data_dir = getattr(cfg, "val_data_dir", train_data_dir)

    assert train_data_dir is not None, "cfg.train_data_dir is required for Waymo training"
    assert Path(train_data_dir).exists(), f"train_data_dir not found: {train_data_dir}"
    assert Path(val_data_dir).exists(), f"val_data_dir not found: {val_data_dir}"

    anchor_path = getattr(cfg, "anchor_path", "data/cluster_64_center_dict.pkl")

    # 可选：只保留必要字段（建议你在 dataset 里支持 keep_keys）
    keep_keys = getattr(cfg, "keep_keys", None)
    strict = bool(getattr(cfg, "strict_keys", False))
    sort_files = bool(getattr(cfg, "sort_files", True))

    train_dataset = WaymoDataset(
        data_dir=train_data_dir,
        keep_keys=keep_keys,
        strict=strict,
        sort_files=sort_files,
    )

    val_dataset = WaymoDataset(
        data_dir=val_data_dir,
        keep_keys=keep_keys,
        strict=strict,
        sort_files=sort_files,
    )

    print(f"[Data] train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collate_fn,   # 关键：让 scenario_id 等字段保留 list
        multiprocessing_context="spawn",  # ✅ 关键
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collate_fn,
        multiprocessing_context="spawn",  # ✅ 关键
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


# ================== init GRPO from IL ckpt ==================
def init_grpo_from_il_ckpt(
    grpo_model: MaskPlannerGRPO,
    il_ckpt_path: Union[str, Path],
):
    """
    从 IL ckpt 初始化 GRPO：
    - 加载 encoder / decoder / mask_net（mask_net 冻结，但仍需加载权重）
    """
    il_ckpt_path = Path(il_ckpt_path)
    assert il_ckpt_path.exists(), f"IL checkpoint not found: {il_ckpt_path}"

    print(f"[GRPO Init] Loading IL checkpoint from: {il_ckpt_path}")
    ckpt = torch.load(il_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    def _strip_prefix(prefix: str, sd: dict):
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    enc_sd = _strip_prefix("encoder.", state_dict)
    dec_sd = _strip_prefix("decoder.", state_dict)
    mask_sd = _strip_prefix("mask_net.", state_dict)

    missing, unexpected = grpo_model.encoder.load_state_dict(enc_sd, strict=False)
    print(f"[GRPO Init] Encoder loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    missing, unexpected = grpo_model.decoder.load_state_dict(dec_sd, strict=False)
    print(f"[GRPO Init] Decoder loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    if hasattr(grpo_model, "mask_net") and len(mask_sd) > 0:
        missing, unexpected = grpo_model.mask_net.load_state_dict(mask_sd, strict=False)
        print(f"[GRPO Init] MaskNet loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[GRPO Init] No mask_net weights found in ckpt (or model has no mask_net).")


# ================== main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/waymo.yaml",
        help="Path to yaml config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["il", "grpo"],
        help="Override training mode (il/grpo). If None, use cfg.train_mode or default il.",
    )
    parser.add_argument(
        "--il_ckpt",
        type=str,
        default=None,
        help="(Optional) IL checkpoint path to init GRPO model.",
    )
    args = parser.parse_args()

    pl.seed_everything(42)

    # ===== 1) cfg =====
    cfg = load_config_from_yaml(args.cfg)
    train_mode = args.mode or getattr(cfg, "train_mode", "il")
    assert train_mode in ["il", "grpo"]
    print(f"[Train] mode = {train_mode}")

    # ===== 2) model =====
    if train_mode == "il":
        model = MaskPlannerMetric(cfg)
        exp_suffix = "IL"
    else:
        model = MaskPlannerGRPO(cfg)
        if args.il_ckpt:
            init_grpo_from_il_ckpt(model, args.il_ckpt)
        else:
            print("[GRPO Init] No IL ckpt provided, GRPO starts from current weights.")
        exp_suffix = "GRPO"

    # ===== 3) loaders =====
    train_loader, val_loader = build_dataloaders(cfg)

    # ===== 4) logger & ckpt =====
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_root = Path(cfg.save_dir) / timestamp
    exp_name = f"{cfg.exp_name}_{exp_suffix}"

    logger = TensorBoardLogger(
        save_dir=str(save_root),
        name=exp_name,
    )

    ckpt_dir = save_root / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 你原模型 val_step log 的 key 是 val_loss（val_{k}），所以 monitor=val_loss
    monitor_key = getattr(cfg, "monitor_key", "val_loss")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_top_k=int(getattr(cfg, "save_top_k", 3)),
        monitor=monitor_key,
        mode=str(getattr(cfg, "monitor_mode", "min")),
        filename="{epoch:02d}-{" + monitor_key + ":.4f}",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ===== 5) trainer =====
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = getattr(cfg, "devices", 1)

    trainer = pl.Trainer(
        max_epochs=int(getattr(cfg, "epochs", 50)),
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=float(getattr(cfg, "grad_clip", 1.0)),
        log_every_n_steps=int(getattr(cfg, "log_every_n_steps", 50)),
        check_val_every_n_epoch=int(getattr(cfg, "check_val_every_n_epoch", 1)),
        enable_checkpointing=True,
    )

    # ===== 6) fit =====
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
