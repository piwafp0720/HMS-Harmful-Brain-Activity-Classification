import itertools
from pathlib import Path

import cv2
import numpy as np
import optimizers
import pandas
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from augmentations import (
    cutmix_data,
    mixup_criterion,
    mixup_data,
    timemix_data,
)
from modules.base_module import BaseModule
from omegaconf import OmegaConf
from torchvision import datapoints
from utils import score


class HMSModule(BaseModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):

        self.timemix_alpha = kwargs.pop("timemix_alpha")
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = []

    def training_step(self, batch, batch_nb):
        if self.ema_decay is not None:
            self.model_ema.update(self.model)
        image, target, n_evaluator = batch

        # self.dump_data(batch_nb, "before_train", image)
        if self.transform_train is not None:
            # image = image.byte()
            image = self.transform_train(image)
        # self.dump_data(batch_nb, "after_train", image, stop_batch_nb=7)

        b_mixed = False
        if np.random.rand() < self.p_mix_augmentation and len(
            self.mix_augmentation
        ):
            b_mixed = True
            method = np.random.choice(self.mix_augmentation)
            if method == "mixup":
                (
                    image,
                    target_a,
                    target_b,
                    lam,
                ) = mixup_data(image, target, self.mixup_alpha)
                y = self.forward(image)
            elif method == "cutmix":
                (
                    image,
                    target_a,
                    target_b,
                    lam,
                ) = cutmix_data(image, target, self.cutmix_alpha)
                y = self.forward(image)
            elif method == "timemix":
                (
                    image,
                    target_a,
                    target_b,
                    lam,
                ) = timemix_data(image, target, self.timemix_alpha)
                y = self.forward(image)
            elif method == "manifold_mixup":
                y, lam, index = self.forward(image, True)
                target_a = target
                target_b = target[index, ...]
            else:
                raise NotImplementedError
        else:
            y = self.forward(image)

        if b_mixed:
            loss_dict, loss = mixup_criterion(
                self.loss,
                y,
                target_a,
                target_b,
                lam,
            )
        else:
            loss_dict, loss = self.loss(y, target)

        loss_dict = {f"train_{k}": v for k, v in loss_dict.items()}

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        image, target, n_evaluator = batch

        # self.dump_data(batch_nb, "before_val", image)
        if self.transform_val is not None:
            # image = image.byte()
            image = self.transform_val(image)
        # self.dump_data(batch_nb, "after_val", image)

        y = self.forward(image)

        loss_dict, loss = self.loss(y, target)
        loss_dict = {f"val_{k}": v for k, v in loss_dict.items()}

        self.log_dict(
            loss_dict,
        )

        output = {
            "y": F.softmax(y, dim=1).detach().cpu().numpy(),
            "target": target.detach().cpu().numpy(),
            "n_evaluator": n_evaluator.detach().cpu().numpy(),
        }
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        metric_dict = {}

        y = np.concatenate(
            [output["y"] for output in self.validation_step_outputs]
        )
        target = np.concatenate(
            [output["target"] for output in self.validation_step_outputs]
        )
        n_evaluator = np.concatenate(
            [output["n_evaluator"] for output in self.validation_step_outputs]
        )

        y[np.isnan(y).sum(axis=1) > 0] = [
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
        ]
        submission = pd.DataFrame(y)
        solution = pd.DataFrame(target)
        submission["id"] = np.arange(len(submission))
        solution["id"] = np.arange(len(solution))

        # score()関数内で上書きされてしまうので、submissionもsolutionもcopy()して使う
        val_metric = score(
            solution=solution.copy(),
            submission=submission.copy(),
            row_id_column_name="id",
        )
        metric_dict["val_metric"] = val_metric

        submission_low = submission.copy()[n_evaluator < 10].reset_index(
            drop=True
        )
        solution_low = solution.copy()[n_evaluator < 10].reset_index(drop=True)
        submission_low["id"] = np.arange(len(submission_low))
        solution_low["id"] = np.arange(len(solution_low))
        val_metric_low = score(
            solution=solution_low,
            submission=submission_low,
            row_id_column_name="id",
        )
        metric_dict["val_metric_low"] = val_metric_low

        submission_high = submission.copy()[n_evaluator >= 10].reset_index(
            drop=True
        )
        solution_high = solution.copy()[n_evaluator >= 10].reset_index(
            drop=True
        )
        submission_high["id"] = np.arange(len(submission_high))
        solution_high["id"] = np.arange(len(solution_high))
        val_metric_high = score(
            solution=solution_high,
            submission=submission_high,
            row_id_column_name="id",
        )
        metric_dict["val_metric_high"] = val_metric_high

        self.log_dict(metric_dict)
        self.validation_step_outputs.clear()

    def _set_refined_augment(self):
        self.transform_train = self.transform_val
        self.datamodule.train_dataset.spec_transforms = (
            self.datamodule.val_dataset.spec_transforms
        )
        self.datamodule.train_dataset.audio_transforms = (
            self.datamodule.val_dataset.audio_transforms
        )
        self.p_mix_augmentation = 0

    def dump_data(
        self,
        batch_nb,
        prefix,
        image,
        label=None,
        stop_batch_nb=None,
    ):
        image = image.detach().cpu().numpy()
        if label is not None:
            label = label.detach().cpu().numpy()
        p_output_root = Path("./tmp/aug_torch/")
        p_output_root.mkdir(exist_ok=True, parents=True)
        filename = f"{prefix}_{batch_nb}.npz"
        np.savez(p_output_root / filename, image=image, label=label)
        if batch_nb == stop_batch_nb:
            exit()
