import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Union

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf

# ====== 模型 ======
from MaskAD.model.maskplanner_metric import MaskPlannerMetric
from MaskAD.GRPO.GRPO_waymo import MaskPlannerGRPO

# ====== Dataset ======
from MaskAD.data_process.dataset import WaymoDataset, data_collate_fn


def load_config_from_yaml(cfg_path: str):
    return OmegaConf.load(cfg_path)


def build_dataloaders(cfg):
    batch_size = int(cfg.batch_size)
    num_workers = int(getattr(cfg, "num_workers", 4))

    train_data_dir = getattr(cfg, "train_data_dir", None)
    val_data_dir = getattr(cfg, "val_data_dir", train_data_dir)

    assert train_data_dir is not None, "cfg.train_data_dir is required"
    assert Path(train_data_dir).exists(), f"train_data_dir not found: {train_data_dir}"
    assert Path(val_data_dir).exists(), f"val_data_dir not found: {val_data_dir}"

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

    # ✅ spawn 对你这种读取 pkl + 多进程更稳
    mp_ctx = "spawn"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collate_fn,               # ✅ 保留 scenario_id/tfrecord_path 为 list
        multiprocessing_context=mp_ctx,
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
        multiprocessing_context=mp_ctx,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def init_grpo_from_il_ckpt(grpo_model: MaskPlannerGRPO, il_ckpt_path: Union[str, Path]):
    il_ckpt_path = Path(il_ckpt_path)
    assert il_ckpt_path.exists(), f"IL checkpoint not found: {il_ckpt_path}"
    print(f"[GRPO Init] Loading IL checkpoint from: {il_ckpt_path}")

    ckpt = torch.load(il_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def pick(prefix: str):
        out = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    enc_sd = pick("encoder.")
    dec_sd = pick("decoder.")
    mask_sd = pick("mask_net.")

    miss, unexp = grpo_model.encoder.load_state_dict(enc_sd, strict=False)
    print(f"[GRPO Init] Encoder loaded. missing={len(miss)}, unexpected={len(unexp)}")

    miss, unexp = grpo_model.decoder.load_state_dict(dec_sd, strict=False)
    print(f"[GRPO Init] Decoder loaded. missing={len(miss)}, unexpected={len(unexp)}")

    if hasattr(grpo_model, "mask_net") and len(mask_sd) > 0:
        miss, unexp = grpo_model.mask_net.load_state_dict(mask_sd, strict=False)
        print(f"[GRPO Init] MaskNet loaded. missing={len(miss)}, unexpected={len(unexp)}")
    else:
        print("[GRPO Init] No mask_net weights found (or model has no mask_net).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        default="/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/waymo.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["il", "grpo"])
    parser.add_argument("--il_ckpt", type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    cfg = load_config_from_yaml(args.cfg)
    train_mode = args.mode or getattr(cfg, "train_mode", "il")
    
    assert train_mode in ["il", "grpo"]
    print(f"[Train] mode = {train_mode}")

    # ===== 1) model =====
    if train_mode == "il":
        model = MaskPlannerMetric(cfg)
        exp_suffix = "IL"
        # ✅ 推荐监控：val_minADE（你按我上面的补丁加了 log 后就会有）
        default_monitor_key = "val_minADE"
        default_monitor_mode = "min"
    else:
        model = MaskPlannerGRPO(cfg)
        if args.il_ckpt:
            init_grpo_from_il_ckpt(model, args.il_ckpt)
        else:
            print("[GRPO Init] No IL ckpt provided, GRPO starts from current weights.")
        exp_suffix = "GRPO"
        # GRPO 你通常会 log reward / return，这里给个默认值（你可在 cfg 里覆盖）
        default_monitor_key = getattr(cfg, "monitor_key", "val_minADE")
        default_monitor_mode = getattr(cfg, "monitor_mode", "min")

    # ===== 2) loaders =====
    train_loader, val_loader = build_dataloaders(cfg)

    # ===== 3) logger & ckpt =====
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_root = Path(cfg.save_dir) / timestamp
    exp_name = f"{cfg.exp_name}_{exp_suffix}"

    logger = TensorBoardLogger(save_dir=str(save_root), name=exp_name)

    ckpt_dir = save_root / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor_key = getattr(cfg, "monitor_key", default_monitor_key)
    monitor_mode = getattr(cfg, "monitor_mode", default_monitor_mode)
    print(f"[CKPT] monitor={monitor_key} mode={monitor_mode}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_top_k=int(getattr(cfg, "save_top_k", 3)),
        monitor=monitor_key,
        mode=monitor_mode,
        filename="{epoch:02d}-{" + monitor_key + ":.4f}",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ===== 4) trainer =====
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = getattr(cfg, "devices", 1)


    trainer = pl.Trainer(
        max_epochs=int(getattr(cfg, "epochs", 50)),
        accelerator=accelerator,
        devices=devices,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=float(getattr(cfg, "grad_clip", 1.0)),
        log_every_n_steps=int(getattr(cfg, "log_every_n_steps", 50)),
        check_val_every_n_epoch=int(getattr(cfg, "check_val_every_n_epoch", 1)),
        enable_checkpointing=True,
        # 可选：precision / accumulate（看你显存）
        precision=getattr(cfg, "precision", "32-true"),
        accumulate_grad_batches=int(getattr(cfg, "accumulate_grad_batches", 1)),
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
