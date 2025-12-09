import torch
import random
import numpy as np
from mmengine import fileio
from pathlib import Path
import io
import os
import json
from types import SimpleNamespace
import lightning.pytorch as pl
from omegaconf import OmegaConf

def save_args_json(cfg: SimpleNamespace, model: pl.LightningModule, log_dir: str):
    """
    模仿 Diffusion-Planner 的行为，把当前训练用到的配置 + normalizer 的统计
    保存为 args.json，方便之后评估 / 推理使用。
    """

    # 1. SimpleNamespace -> dict
    cfg_dict = vars(cfg).copy()

    # 2. normalizer，如有则加入
    state_norm = getattr(model, "state_normalizer", None)
    if state_norm is not None and hasattr(state_norm, "to_dict"):
        cfg_dict["state_normalizer"] = state_norm.to_dict()

    obs_norm = getattr(model, "obs_normalizer", None)
    if obs_norm is not None and hasattr(obs_norm, "to_dict"):
        cfg_dict["observation_normalizer"] = obs_norm.to_dict()

    # 3. 使用 OmegaConf 的自动转换功能（最稳的方法）
    cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)

    # 4. 保存 JSON
    args_path = Path(log_dir) / "args.json"
    args_path.parent.mkdir(parents=True, exist_ok=True)

    with args_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4, ensure_ascii=False)

    print(f"[Config] args.json saved to: {args_path}")



def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)

    return npz_data

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
    """
    save the model to path
    """
    save_model = {'epoch': epoch + 1, 
                  'model': model.state_dict(), 
                  'ema_state_dict': ema.state_dict(),
                  'optimizer': optimizer.state_dict(), 
                  'schedule': scheduler.state_dict(), 
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    with io.BytesIO() as f:
        torch.save(save_model, f)
        fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")

def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = fileio.get(path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f)

    # load model
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict(ckpt)                   
    print("Model load done")
    
    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")
            
    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")
    
    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
