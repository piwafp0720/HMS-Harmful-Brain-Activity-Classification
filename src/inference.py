import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path

import cupy as cp
import cv2
import models
import numpy as np
import pandas
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
import yaml
from augmentations import *
from mydatasets import *
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score
from tqdm import tqdm
from utils import score
from utils.general_utils import omegaconf_to_yaml, parser

target_columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
prediction_columns = [f"proba_{c}" for c in target_columns]
logit_columns = [f"logit_{c}" for c in target_columns]


def load_net(ckpt_path, device):
    cfg_path = Path(ckpt_path).parents[1] / "train_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    args = cfg["model"]["args"]

    args["pretrained"] = False

    net = getattr(models, cfg["model"]["name"])(**args)
    net.to(device)
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{device}")["state_dict"]
    ckpt = {k[k.find(".") + 1 :]: v for k, v in ckpt.items()}

    missing_keys, unexpected_keys = net.load_state_dict(ckpt, strict=False)
    assert not missing_keys, f"{ckpt_path=}\n{missing_keys=}"
    net.eval()
    net.config = cfg

    return net


def get_loader(
    cfg,
    csv_path,
    transforms,
    batch_size,
    num_workers,
    dry_run,
    mode="test",
):
    tmp_cfg = cfg["dataset"].copy()
    specs = np.load(cfg["specs_path"], allow_pickle=True).item()

    dataset_args = {
        **tmp_cfg,
        "specs": specs,
        "mode": mode,
        "spec_transforms": transforms.spec_transform_test,
        "audio_transforms": transforms.audio_transform_test,
        "dry_run": dry_run,
    }

    if csv_path is not None:
        dataset_args["csv_path"] = csv_path

    if mode == "test":
        dataset_args["fold"] = None

    if mode == "pseudo_label":
        dataset_args["group"] = None

    dataset = eval(cfg["dataset_name"])(**dataset_args)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


def apply_tta(net, image, i_tta):
    if i_tta == 0:
        pred = net(image)
    elif i_tta == 1:  # flip width
        pred = net(image.flip(-1))
        raise ValueError
    return pred


def inference_core(loader, transform, ckpt_list, mode, device, n_tta, use_amp):
    net_list = []
    for ckpt in ckpt_list:
        net_list.append(load_net(ckpt, device))

    results = []
    results_logit = []
    for i, batch in enumerate(tqdm(loader, desc=f"prediction")):
        image, target, n_evaluator = batch

        image = image.to(device)
        if transform is not None:
            # image = image.byte()
            image = transform(image)

        result_net = []
        result_net_logit = []
        for net in net_list:
            result_tta = []
            result_tta_logit = []
            for i_tta in range(n_tta):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    if net.config["dataset"]["order"] == "tile":
                        _, _, h, w = image.shape
                        resize_h = net.config["dataset"]["resize_h"]
                        resize_w = net.config["dataset"]["resize_w"]
                        if h != resize_h or w != resize_w:
                            image = F.interpolate(
                                image, (resize_h, resize_w), mode="bilinear"
                            )
                    pred = apply_tta(net, image, i_tta)
                result_tta_logit.append(pred.float().detach().cpu().numpy())
                pred = F.softmax(pred.float(), dim=1).detach().cpu().numpy()
                result_tta.append(pred)
            result_tta = np.array(result_tta).mean(axis=0)
            result_tta_logit = np.array(result_tta_logit).mean(axis=0)
            result_net.append(result_tta)
            result_net_logit.append(result_tta_logit)
        result_net = np.array(result_net).mean(axis=0)
        result_net_logit = np.array(result_net_logit).mean(axis=0)
        # ensemble結果のラベルの合計が1になるようにする
        result_net = result_net / result_net.sum(axis=1, keepdims=True)
        results.append(result_net)
        results_logit.append(result_net_logit)
    results = np.concatenate(results)
    results_logit = np.concatenate(results_logit)
    return results, results_logit


def calc_metric(df, prediction_columns):
    df[prediction_columns] = df[prediction_columns].astype("float32")
    df[target_columns] = df[target_columns].astype("float32")
    submission = df[prediction_columns]
    submission.columns = target_columns
    solution = df[target_columns]
    submission["id"] = np.arange(len(submission))
    solution["id"] = np.arange(len(solution))
    metric = score(
        solution=solution, submission=submission, row_id_column_name="id"
    )
    metric_dict = {"metric": metric}
    return metric_dict


def inference(config):
    mode = config["mode"]
    assert mode in ["val", "test", "pseudo_label"]

    csv_path = config["csv_path"]
    device = config["gpu"]
    batch_size = config["batch_size"]
    n_tta = config["n_tta"]
    seed = config["seed"]
    num_workers = config["n_workers"]
    use_amp = config["use_amp"]
    dry_run = config["dry_run"]

    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    output_dir = Path(config["save_root"]) / config["version"]

    if output_dir.exists():
        print("This version already exists.\n" f"version:{output_dir}")
        ans = None
        while ans not in ["y", "Y"]:
            ans = input("Do you want to continue inference? (y/n): ")
            if ans in ["n", "N"]:
                quit()
    output_dir.mkdir(exist_ok=True, parents=True)

    cfg_path = Path(config["checkpoint"][0]).parents[1] / "train_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    transforms = eval(cfg["augmentation_name"])(cfg["augmentation_ver"])
    loader = get_loader(
        cfg,
        csv_path,
        transforms,
        batch_size,
        num_workers,
        dry_run,
        mode,
    )
    df = loader.dataset.df

    # torch_transform = transform.torch_test
    torch_transform = None

    with torch.no_grad():
        results, results_logit = inference_core(
            loader,
            torch_transform,
            config["checkpoint"],
            mode,
            device,
            n_tta,
            use_amp,
        )

    # nan除け
    results[np.isnan(results).sum(axis=1) > 0] = [
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
    ]
    results_logit[np.isnan(results_logit).sum(axis=1) > 0] = [
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
    ]

    # save probabilities & logits
    df[prediction_columns] = results
    df[logit_columns] = results_logit
    df.to_csv(output_dir / f"prediction.csv", index=False)

    # save submission file
    submission = df[["eeg_id"]].copy()
    submission[target_columns] = results
    submission.to_csv(output_dir / f"submission.csv", index=False)

    if mode != "test":
        metric_dict = calc_metric(df, prediction_columns)
        metric_df = pd.Series(metric_dict)

        metric_df.to_csv(output_dir / "scores.csv")
        print("#" * 20)
        print(metric_df)

    with open(output_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f.name)
    with open(output_dir / "inference.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config=config, f=f.name)


def get_cv_score(config, version):
    p_fold_root = Path(config["save_root"]) / version
    p_fold_list = list(p_fold_root.glob("fold_*"))

    result_dfs = []
    for p_fold in p_fold_list:
        result_df = pd.read_csv(p_fold / "prediction.csv")
        result_dfs.append(result_df)

    result_dfs = pd.concat(result_dfs).reset_index(drop=True)
    metric_dict = calc_metric(result_dfs, prediction_columns)
    metric_df = pd.Series(metric_dict)
    result_dfs.to_csv(p_fold_root / "result_df.csv", index=False)
    metric_df.to_csv(p_fold_root / "scores.csv")
    print("#" * 20)
    print(metric_df)


def merge_oof(config, version):
    p_fold_root = Path(config["save_root"]) / version
    p_fold_list = list(p_fold_root.glob("fold_*"))

    prediction_dfs = []
    for p_fold in p_fold_list:
        prediction_df = pd.read_csv(p_fold / "prediction.csv")
        prediction_dfs.append(prediction_df)

    submission_dfs = []
    for p_fold in p_fold_list:
        submission_df = pd.read_csv(p_fold / "submission.csv")
        submission_dfs.append(submission_df)

    prediction_dfs = pd.concat(prediction_dfs).reset_index(drop=True)
    prediction_dfs.to_csv(p_fold_root / "prediction.csv", index=False)
    submission_dfs = pd.concat(submission_dfs).reset_index(drop=True)
    submission_dfs.to_csv(p_fold_root / "submission.csv", index=False)


def main():
    cfg = parser("test")

    tta = cfg["n_tta"]
    version = cfg["version"]
    mode = cfg["mode"]
    # assert tta == 1

    org_checkpoint_list = cfg["checkpoint"]
    k_fold = len(org_checkpoint_list)

    version = f"{version}/tta_{tta}"
    if mode == "val":
        for fold, checkpoint in enumerate(org_checkpoint_list):
            cfg["version"] = version + f"/fold_{fold}"
            cfg["checkpoint"] = checkpoint
            inference(cfg)

        if len(org_checkpoint_list) > 1:
            get_cv_score(cfg, version)
    elif mode == "pseudo_label":
        for fold, checkpoint in enumerate(org_checkpoint_list):
            cfg["version"] = version + f"/fold_{fold}"
            cfg["checkpoint"] = checkpoint
            inference(cfg)
        merge_oof(cfg, version)
    elif mode == "test":
        cfg["version"] = version
        inference(cfg)


if __name__ == "__main__":
    main()
