import re
import shutil
import sys
import time
from pathlib import Path

import albumentations
import losses
import models
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.general_utils import get_hms


def train(
    cfg,
    datamodule,
    dataset_args,
    module,
    ckpt_name="{epoch:02}-{val_loss:.3f}",
    module_kwargs={},
):

    assert cfg.logger.mode in ["online", "offline", "debug"]
    gpu = cfg.gpu
    torch.cuda.set_device(gpu)
    seed = np.random.randint(65535) if cfg.seed is None else cfg.seed
    seed_everything(seed)

    if cfg.logger.mode == "debug":
        logger = False
        callbacks = []
    else:
        logger = WandbLogger(
            name=cfg.logger.runName,
            project=cfg.logger.project,
            offline=True if cfg.logger.mode == "offline" else False,
            log_model=False,
        )
        logger.experiment.config.update(dict(cfg))
        logger.experiment.config.update({"dir": logger.experiment.dir})

        ckpt_dir = Path(logger.experiment.dir) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=ckpt_name,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            save_weights_only=cfg.checkpoint.save_weights_only,
            every_n_epochs=cfg.checkpoint.every_n_epochs,
            verbose=True,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
        )
        lr_monitor = LearningRateMonitor(logging_interval=None)
        callbacks = [checkpoint, lr_monitor]

        OmegaConf.save(cfg, ckpt_dir.parent / "train_config.yaml")

        augmentation_file_path = (
            Path(__file__).parent / "augmentations/augmentation.py"
        )
        if augmentation_file_path.exists():
            shutil.copy(augmentation_file_path, logger.experiment.dir)

    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        max_steps=cfg.train.step if "step" in cfg.train.keys() else -1,
        accumulate_grad_batches=cfg.train.n_accumulations,
        limit_val_batches=1.0,
        val_check_interval=cfg.train.val_check_interval,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        devices=[gpu],
        callbacks=callbacks,
        precision=16 if cfg.train.amp else 32,
        log_every_n_steps=1,
        gradient_clip_val=cfg.train.gradient_clip_val,
        deterministic=cfg.train.deterministic,
    )
    torch.use_deterministic_algorithms(cfg.train.deterministic)

    net = getattr(models, cfg.model.name)(**cfg.model.args)

    if cfg.model.load_checkpoint is not None:
        ckpt = torch.load(
            cfg.model.load_checkpoint, map_location=f"cuda:{gpu}"
        )["state_dict"]
        ckpt = {k[k.find(".") + 1 :]: v for k, v in ckpt.items()}
        missing_keys, unexpected_keys = net.load_state_dict(ckpt, strict=False)
        print(f"\nload checkpoint: {cfg.model.load_checkpoint}\n")

    if cfg.loss is not None:
        if "name" in cfg.loss.keys():
            loss_args = (
                dict(cfg.loss.args)
                if "args" in cfg.loss.keys() and cfg.loss.args is not None
                else {}
            )
            for k, v in loss_args.items():
                if k in ["weight", "pos_weight"]:
                    loss_args[k] = torch.tensor(v)
            loss = getattr(losses, cfg.loss.name)(**loss_args)
        else:
            loss = {}
            for loss_key, loss_dict in cfg.loss.items():
                loss_args = (
                    dict(loss_dict.args)
                    if "args" in cfg.loss.keys() and cfg.loss.args is not None
                    else {}
                )
                for k, v in loss_args.items():
                    if k in ["weight", "pos_weight"]:
                        loss_args[k] = torch.tensor(v)
                loss[loss_key] = getattr(losses, loss_dict.name)(**loss_args)
    else:
        loss = None

    data_module = datamodule(
        train_args=dataset_args["train"],
        val_args=dataset_args["val"],
        **cfg.datamodule,
    )

    model = module(
        model=net,
        loss=loss,
        optimizer_name=cfg.optimizer.name,
        optimizer_args=cfg.optimizer.args,
        scheduler=cfg.optimizer.scheduler.name,
        scheduler_args=cfg.optimizer.scheduler.args,
        lr_dict_param=cfg.optimizer.scheduler.lr_dict_param,
        freeze_start=cfg.model.freeze_start.target_epoch,
        unfreeze_params=cfg.model.freeze_start.unfreeze_params,
        **module_kwargs,
    )

    model.datamodule = data_module

    sec_train_start = time.time()
    trainer.fit(model, data_module)
    sec_train_end = time.time()

    if cfg.logger.mode != "debug":
        # スコアが良い順にwandbにpathを記録
        def extract_float_from_filename(filename):
            """ファイル名から浮動小数点数を抽出するヘルパー関数"""
            # match = re.search(r"=(\d+\.\d+)\.ckpt", filename)
            match = re.search(
                rf"{cfg['checkpoint']['monitor']}=(\d+\.\d+)", filename
            )
            if match:
                return float(match.group(1))
            else:
                float_max = sys.float_info.max
                if cfg.checkpoint.mode == "max":
                    return -float_max
                elif cfg.checkpoint.mode == "min":
                    return float_max

        ckpt_list = [
            str(p.resolve())
            for p in Path(logger.experiment.dir).glob("**/*.ckpt")
        ]
        reverse = True if cfg.checkpoint.mode == "max" else False
        ckpt_list = sorted(
            ckpt_list, key=extract_float_from_filename, reverse=reverse
        )

        for n, model_path in enumerate(ckpt_list):
            name = f"model_{n}"
            logger.experiment.config.update({name: model_path})

        # with open("./wandb/filepath_list.txt", "a") as f:
        #     filename = ckpt_list[0]
        #     match = re.search(r"=(\d+\.\d+)\.ckpt", filename)
        #     score = float(match.group(1))
        #     h, m, s = get_hms(sec_train_end - sec_train_start)
        #     training_time = f"{h}h{m}m{s}s"
        #     f.write(
        #         f"{cfg.logger.runName} {score} {training_time} {filename}\n"
        #     )

        with open("./wandb/filepath_list.txt", "a") as f:
            filename = ckpt_list[0]
            matches = re.findall(
                r"val_metric(?:_low|_high)?=(\d+\.\d+)", filename
            )
            scores = " ".join(matches)
            h, m, s = get_hms(sec_train_end - sec_train_start)
            training_time = f"{h}h{m}m{s}s"
            f.write(
                f"{cfg.logger.runName} {scores} {training_time} {filename}\n"
            )
