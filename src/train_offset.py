import copy
import math
import os
import re
from pathlib import Path

import models
import numpy as np
import pandas as pd
import torch
from augmentations.augmentation import *
from base_train import train
from modules import *
from mydatasets import *
from torch.utils.data import DataLoader
from utils.general_utils import parser


def main():
    cfg = parser("train")

    if Path(cfg["dataset"]["csv_path"]).name == "fold_pseudo_label.csv":
        assert cfg["dataset"]["label_smoothing_ver"] == None

    if cfg["dataset"]["group"] == "high_vote":
        cfg["checkpoint"]["monitor"] = "val_metric_high"

    specs = np.load(cfg["specs_path"], allow_pickle=True).item()

    datamodule = eval(cfg["datamodule_name"])

    transforms = eval(cfg["augmentation_name"])(cfg["augmentation_ver"])
    dataset_args = {
        "train": {
            "specs": specs,
            "mode": "train",
            "spec_transforms": transforms.spec_transform_train,
            "audio_transforms": transforms.audio_transform_train,
            **cfg.dataset,
        },
        "val": {
            "specs": specs,
            "mode": "val",
            "spec_transforms": transforms.spec_transform_val,
            "audio_transforms": transforms.audio_transform_val,
            **cfg.dataset,
        },
    }

    module = eval(cfg["module_name"])
    module_kwargs = dict(cfg.module) if cfg.module is not None else {}
    module_kwargs.update(
        {
            # "transform_train": transforms.torch_train,
            # "transform_val": transforms.torch_val,
        }
    )

    if isinstance(cfg["train"]["epoch"], float):
        dataset = eval(cfg["dataset_name"])
        len_dataset = len(dataset(**dataset_args["train"]))
        batch_size = cfg["datamodule"]["batch_size"]
        n_accumulations = cfg["train"]["n_accumulations"]
        n_iter_per_epoch = math.ceil(
            len_dataset / batch_size / n_accumulations
        )
        cfg["train"]["step"] = int(n_iter_per_epoch * cfg["train"]["epoch"])
        cfg["train"]["epoch"] = None
    else:
        cfg["train"]["step"] = -1

    if cfg["loss"]["name"] == "CrossEntropyLoss":
        if "weight" in cfg["loss"]["args"].keys() and isinstance(
            cfg["loss"]["args"]["weight"], str
        ):
            cfg["loss"]["args"]["weight"] = np.load(
                cfg["loss"]["args"]["weight"]
            ).tolist()
    elif cfg["loss"]["name"] == "FocalLoss":
        if "alpha" in cfg["loss"]["args"].keys() and isinstance(
            cfg["loss"]["args"]["alpha"], str
        ):
            cfg["loss"]["args"]["alpha"] = np.load(
                cfg["loss"]["args"]["alpha"]
            ).tolist()

    ckpt_name = f"{{epoch:02}}-{{step}}-{{val_metric_low:.4f}}-{{val_metric_high:.4f}}-{{val_metric:.4f}}"
    train(cfg, datamodule, dataset_args, module, ckpt_name, module_kwargs)


if __name__ == "__main__":
    main()
