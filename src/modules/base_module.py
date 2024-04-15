from typing import Union

import optimizers
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from timm.utils import ModelEmaV2
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: Union[nn.Module, dict],
        optimizer_name: Union[str, None] = None,
        optimizer_args: dict = {},
        scheduler: Union[str, None] = None,
        scheduler_args: dict = {},
        lr_dict_param: dict = {},
        freeze_start: Union[int, None] = None,
        unfreeze_params: Union[list, None] = None,
        transform_train: Union[T.Compose, None] = None,
        transform_val: Union[T.Compose, None] = None,
        p_transform: float = 0.5,
        mix_augmentation: Union[list, None] = None,
        p_mix_augmentation: float = 0.0,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        last_n_epoch_refined_augment: int = -1,
        ema_decay: Union[float, None] = None,
    ):
        super().__init__()
        self.model = model
        self.ema_decay = ema_decay
        if ema_decay is not None:
            self.model_ema = ModelEmaV2(self.model, decay=ema_decay)
        self.loss = loss
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.lr_dict_param = lr_dict_param
        self.freeze_start = freeze_start
        self.unfreeze_params = unfreeze_params
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.p_transform = p_transform
        self.mix_augmentation = mix_augmentation
        self.p_mix_augmentation = p_mix_augmentation
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.last_n_epoch_refined_augment = last_n_epoch_refined_augment

    def forward(self, *x):
        if self.ema_decay is not None:
            y = self.model_ema.module(*x)
        else:
            y = self.model(*x)
        return y

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)

        loss = self.loss(y, t)

        logger_logs = {"loss": loss}

        output = {"loss": loss, "progress_bar": {}, "log": logger_logs}
        return output

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)

        val_batch_loss = self.loss(y, t)

        output = {
            "val_batch_loss": val_batch_loss,
        }
        return output

    def configure_optimizers(self):
        if OmegaConf.is_list(self.optimizer_args):
            kwargs_dict = {}
            for kwargs in self.optimizer_args:
                param_names = kwargs.pop("params")
                if param_names == "default":
                    default_kwargs = kwargs
                else:
                    if isinstance(param_names, str):
                        param_names = [param_names]

                    for param in param_names:
                        kwargs_dict[param] = kwargs

            optimized_params = []
            for n, p in self.model.named_parameters():
                for i, (param, kwargs) in enumerate(kwargs_dict.items()):
                    if param in n:
                        optimized_params.append({"params": p, **kwargs})
                        break
                    elif i == len(kwargs_dict) - 1:
                        optimized_params.append(
                            {
                                "params": p,
                            }
                        )

            optimizer = getattr(optimizers, self.optimizer_name)(
                optimized_params, **default_kwargs
            )

        elif OmegaConf.is_dict(self.optimizer_args):
            optimizer = getattr(optimizers, self.optimizer_name)(
                self.parameters(), **self.optimizer_args
            )
        else:
            raise TypeError

        if self.scheduler is None:
            return optimizer
        else:
            self._update_scheduler_settings()
            scheduler = getattr(optimizers, self.scheduler)(
                optimizer, **self.scheduler_args
            )
            if self.lr_dict_param:
                scheduler = {"scheduler": scheduler, **self.lr_dict_param}
            return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        if (
            self.freeze_start
            and self.unfreeze_params
            and self.current_epoch == 0
        ):
            self.freeze()
            print("==Freeze Start==")
            print("Unfreeze params:")
            for n, p in self.model.named_parameters():
                if any(param in n for param in self.unfreeze_params):
                    p.requires_grad = True
                    print(f"  {n}")
            self.model.train()

        if self.current_epoch == self.freeze_start:
            print("==Unfreeze==")
            self.unfreeze()

        if (
            self.current_epoch
            == self.trainer.max_epochs - self.last_n_epoch_refined_augment
        ):
            self._set_refined_augment()

    def _set_refined_augment(self):
        self.transform_train = self.transform_val
        self.p_mix_augmentation = 0

    def _update_scheduler_settings(self):
        if self.lr_dict_param:
            if (
                "interval" in self.lr_dict_param.keys()
                and self.lr_dict_param.interval == "step"
            ):
                # specify steps if stepwise lr scheduling.
                n_iter = self.trainer.estimated_stepping_batches
                print("n_iter:", n_iter)
                if self.scheduler == "CosineAnnealingLR":
                    self.scheduler_args.T_max = n_iter
                elif self.scheduler == "CosineAnnealingWarmupRestarts":
                    warmup_steps = self.scheduler_args.warmup_steps
                    assert warmup_steps == 0 or isinstance(warmup_steps, float)
                    self.scheduler_args.first_cycle_steps = n_iter
                    self.scheduler_args.warmup_steps = int(
                        warmup_steps * n_iter
                    )
            else:
                if self.scheduler == "CosineAnnealingWarmupRestarts":
                    assert isinstance(self.scheduler_args.warmup_steps, int)
