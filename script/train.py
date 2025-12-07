import torch
import yaml
import datetime
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore tensorflow warnings

# set tf to cpu only
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import jax
jax.config.update("jax_platform_name", "cpu")

from data.dataset import WaymaxDataset
from DPR.model.DPR import DPR
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger


from matplotlib import pyplot as plt



def train(cfg):
    print("Start Training")  # 打印开始训练的提示信息

    # 设置随机种子以确保结果的可重复性
    pl.seed_everything(cfg["seed"])
    # 设置浮点数矩阵乘法的精度为高
    torch.set_float32_matmul_precision("high")    

    # 创建训练数据集
    train_dataset = WaymaxDataset(
        data_dir=cfg["train_data_path"],  # 训练数据路径
        anchor_path=cfg["anchor_path"],  # 锚点路径
        # max_object= cfg["agents_len"],  # 可选参数，最大对象数
    )

    # 创建验证数据集
    val_dataset = WaymaxDataset(
        cfg["val_data_path"],  # 验证数据路径
        anchor_path=cfg["anchor_path"],  # 锚点路径
        # max_object= cfg["agents_len"],  # 可选参数，最大对象数
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"],  # 批量大小
        pin_memory=True,  # 是否固定内存数据会被加载到固定内存（pinned memory）中，这可以加速数据从 CPU 传输到 GPU 的过程。
        num_workers=cfg["num_workers"],  # 加载数据时候的工作线程数
        shuffle=True  # 是否打乱数据
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["batch_size"],  # 批量大小
        pin_memory=True,  # 是否固定内存
        num_workers=cfg["num_workers"],  # 工作线程数
        shuffle=False  # 验证数据不打乱
    )

    # 设置输出路径
    output_root = cfg.get("log_dir", "output")  # 日志目录 cfg.get("log_dir", "output") 表示尝试从配置中获取键 "log_dir" 的值。如果该键不存在，则使用默认值 "output"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 当前时间戳格式为 年-月-日-时-分-秒（例如：20231005123045）
    model_name = f"{cfg['model_name']}_{timestamp}"  # 模型名称  例如，如果 cfg['model_name'] 是 "resnet50"，则 model_name 可能是 "resnet50_20231005123045"。
    output_path = f"{output_root}/{model_name}"  # 完整输出路径  例如，如果 output_root 是 "output"，model_name 是 "resnet50_20231005123045"，则 output_path 是 "output/resnet50_20231005123045"。
    print("Save to ", output_path)  # 打印保存路径

    os.makedirs(output_path, exist_ok=True)  # 创建输出目录
    # 将配置保存为 YAML 文件
    with open(f"{output_path}/config.yaml", "w") as file:  # 打开保存路径下的配置文件
        yaml.dump(cfg, file)  # 将配置字典写入 YAML 文件

    num_gpus = torch.cuda.device_count() # 获取可用的 GPU 数量
    print("Total GPUS:", num_gpus)  # 打印 GPU 数量
    model = DPR(cfg=cfg)  # 初始化模型

    # 如果提供了检查点路径，则加载权重
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print("Load Weights from ", ckpt_path)  # 打印加载权重路径
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"])

    # 如果不训练编码器，则加载编码器权重
    # 需要从命令行传输
    if not cfg.get("train_encoder"):
        encoder_path = cfg.get("encoder_ckpt", None)  # 编码器检查点路径
        if encoder_path is not None:
            model_dict = torch.load(encoder_path, map_location=torch.device("cpu"))["state_dict"]
            for key in list(model_dict.keys()):
                if not key.startswith("encoder."):  # 只保留编码器相关的权重，如果键名不以 "encoder." 开头，则删除对应的权重。
                    del model_dict[key]
            print("Load Encoder Weights")  # 打印加载编码器权重信息
            model.load_state_dict(model_dict, strict=False)  # 加载权重
        else:
            cfg["train_encoder"] = True
            raise Warning("Encoder path is not provided")  # 如果未提供编码器路径，抛出警告

    # plt.plot(model.noise_scheduler.alphas_cumprod.cpu().numpy())
    # plt.plot(f"{output_path}/scheduler.jpg")  # 保存调度器曲线
    # plt.savefig(f"{output_path}/scheduler.jpg")  # 保存调度器曲线图像
    # plt.close()

    use_wandb = cfg.get("use_wandb", True)
    loggers = []

    # TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=output_path,   # 日志根目录
        name="tb",              # 子目录名，最终路径类似：output/.../tb/version_0/
        default_hp_metric=False # 关闭默认的超参指标
    )
    loggers.append(tb_logger)

    # 可选：继续用 wandb
    if use_wandb:
        loggers.append(WandbLogger(
            name=model_name,
            project=cfg.get("project"),
            entity=cfg.get("username"),
            log_model=False,
            dir=output_path,
        ))
    else:
        loggers.append(CSVLogger(output_path, name="DPR", version=1, flush_logs_every_n_steps=100))

    trainer = pl.Trainer(
        num_nodes=cfg.get("num_nodes", 1),
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
        enable_progress_bar=True,
        logger=loggers,             # 这里传入列表即可同时记录到多个日志器
        enable_model_summary=True,
        detect_anomaly=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        log_every_n_steps=1,
        devices=[0], 
        callbacks=[
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=20,
                save_weights_only=False,
                monitor="val/loss",
                filename="epoch={epoch:02d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step")
        ]
    )

    print("Build Trainer success")  # 打印训练器构建完成信息

    # 开始训练
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.get("init_from"))
    # trainer.fit(model, train_loader, val_loader, ckpt_path='/home/ubuntu/LTL/My_Python/AAAI2025/train_DPR/DPR_20250606064949/epoch=04.ckpt')



def load_config(file_path):
    with open(file_path, "r") as file:  # 打开指定路径的 YAML 文件
        data = yaml.safe_load(file)  # 使用安全加载方法读取 YAML 文件内容
    return data  # 返回加载的配置数据

def build_parser():
    '''此处传的参数会覆盖DPR.yaml中的参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str, default="/home/ubuntu/LTL/My_Python/AAAI2025/config/DPR.yaml")
    
    # Params for override config
    parser.add_argument("-name", "--model_name", type=str, default=None)
    parser.add_argument("-log", "-log_dir", type=str, default=None)
    
    parser.add_argument("-step", "--diffusion_steps", type=int, default=None)
    parser.add_argument("-mean", "--action_mean", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-std", "--action_std", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-zD", "--embeding_dim", type=int, default=5)
    parser.add_argument("-clamp", "--clamp_value", type=float, default=None)
    parser.add_argument("-init", "--init_from", type=str, default=None)
    parser.add_argument("-encoder", "--encoder_ckpt", type=str, default=None)
    parser.add_argument("-nN", "--num_nodes", type=int, default=1)
    parser.add_argument("-nG", "--num_gpus", type=int, default=-1)
    parser.add_argument("-sType", "--schedule_type", type=str, default=None)
    parser.add_argument("-sS", "--schedule_s", type=float, default=None)
    parser.add_argument("-sE", "--schedule_e", type=float, default=None)
    parser.add_argument("-scale", "--schedule_scale", type=float, default=None)
    parser.add_argument("-sT", "--schedule_tau", type=float, default=None)
    parser.add_argument("-eV", "--encoder_version", type=str, default=None)
    parser.add_argument("-pred", "--with_predictor", type=bool, default=None)
    parser.add_argument("-type", "--prediction_type", type=str, default=None)
    
    return parser
    
def load_cfg(args):
    """
    加载并更新配置文件

    参数:
        args (argparse.Namespace): 命令行参数对象

    返回:
        dict: 更新后的配置字典
    """
    # 加载配置文件内容到字典
    cfg = load_config(args.cfg)
    
    # 从命令行参数中覆盖配置文件中的配置
    # 遍历命令行参数并更新配置字典
    for key, value in vars(args).items():
        if key == "cfg":  # 跳过配置文件路径参数
            pass
        elif value is not None:  # 如果参数值不为空，则覆盖配置
            cfg[key] = value
    return cfg  # 返回更新后的配置字典
    
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)
    
    train(cfg)
    
