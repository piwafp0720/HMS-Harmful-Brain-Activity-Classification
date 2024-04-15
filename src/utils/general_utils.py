import argparse
import contextlib
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Optional

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf, dictconfig, listconfig
from tqdm.auto import tqdm


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time()-t0:.2f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def omegaconf_to_yaml(cfg):
    if isinstance(cfg, dictconfig.DictConfig):
        cfg = dict(cfg)
        for key, value in cfg.items():
            cfg[key] = omegaconf_to_yaml(value)
    elif isinstance(cfg, listconfig.ListConfig):
        cfg = list(cfg)
        for value in cfg:
            value = omegaconf_to_yaml(value)
    else:
        return cfg
    return cfg


def get_hms(sec):
    hour = int(sec / (60 * 60))
    minute = int((sec % (60 * 60)) / 60)
    second = int((sec % (60 * 60)) % 60)
    return hour, minute, second


def mkdir_with_confirmation(output_dir):
    if output_dir.exists():
        print("This version already exists.\n" f"version:{output_dir}")
        ans = None
        while ans not in ["y", "Y"]:
            ans = input("Do you want to continue inference? (y/n): ")
            if ans in ["n", "N"]:
                quit()
    output_dir.mkdir(exist_ok=True, parents=True)


@contextlib.contextmanager
def tqdm_joblib(total, **kwargs):
    """
    https://yururi-do.com/use-joblib-and-tqdm-to-display-progress-bar-at-batch-level/
    """
    progress_bar = tqdm(total=total, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallBack(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)

            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack

    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress_bar.close()


def allkeys(x):
    for key, value in x.items():
        yield key
        if isinstance(value, dict):
            for child in allkeys(value):
                yield key + "." + child


def check_dotlist(cfg, dotlist):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_keys = list(allkeys(cfg_dict))
    dotlist_dict = OmegaConf.to_container(dotlist, resolve=True)
    dotlist_keys = list(allkeys(dotlist_dict))

    for d_key in dotlist_keys:
        assert d_key in cfg_keys, f"{d_key} dosen't exist in config file."


def make_map_anchor_to_value(yaml_path):
    with open(yaml_path, "r") as file:
        yaml_content = file.read()

    # アンカー名とその値を格納する辞書
    map_anchor_to_value = {}

    for line in yaml_content.split("\n"):
        # コメントは無視する
        if line.strip().startswith("#"):
            pass
        # アンカーをチェック
        elif "&" in line:
            key, value = line.split(":")
            anchor_name = value.split("&")[1].strip().split(" ")[0]
            value_without_anchor = value.replace(f"&{anchor_name}", "").strip()
            map_anchor_to_value[anchor_name] = value_without_anchor
    return map_anchor_to_value


def overwrite_anchor_value(options, map_anchor_to_value):
    for option in options:
        for k in map_anchor_to_value.keys():
            if k in option:
                new_value = option.split("=")[1]
                map_anchor_to_value[k] = new_value
                continue
    return map_anchor_to_value


def resolve_yaml_anchors_and_aliases(yaml_path, map_anchor_to_value):
    with open(yaml_path, "r") as file:
        yaml_content = file.read()

    # 解決された行を格納するリスト
    resolved_lines = []

    for line in yaml_content.split("\n"):
        # コメントは無視する
        if line.strip().startswith("#"):
            resolved_lines.append(line)
        # アンカーを新しい値に置換
        elif "&" in line:
            key, value = line.split(":")
            anchor_name = value.split("&")[1].strip().split(" ")[0]
            value = map_anchor_to_value[anchor_name]
            resolved_lines.append(f"{key}: {value}")
        # エイリアスをチェック
        elif "*" in line:
            key, value = line.split(":")
            alias_name = value.split("*")[1]
            if alias_name in map_anchor_to_value:
                resolved_line = f"{key}: {map_anchor_to_value[alias_name]}"
                resolved_lines.append(resolved_line)
            else:
                raise ValueError(
                    f"alias *{alian_name} doesn't exist in anchors."
                )
        else:
            resolved_lines.append(line)

    return "\n".join(resolved_lines)


def parser(mode):
    assert mode in ["train", "test"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        required=True,
        help="path of the config file",
    )
    parser.add_argument(
        "--options", "-o", nargs="*", help="optional arguments"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)

    # yamlファイルに存在しないパラメーターがdotlistに指定されていないかチェック
    if args.options is not None:
        dotlist = OmegaConf.from_dotlist(args.options)
        check_dotlist(cfg, dotlist)

    # anchor/aliasを手動で解決
    # dotlistによるanchorの上書きを可能にする
    if args.options is not None:
        # anchorとその値を保持する辞書を作成
        map_anchor_to_value = make_map_anchor_to_value(args.config_file)
        # dotlistに含まれるanchorの値で上書きする
        map_anchor_to_value = overwrite_anchor_value(
            args.options, map_anchor_to_value
        )
        # anchor/aliasを解決したyamlの文字列を生成
        yaml_string = resolve_yaml_anchors_and_aliases(
            args.config_file, map_anchor_to_value
        )
        # OmegaConfでconfig化
        cfg = OmegaConf.create(yaml_string)
        # dotlistのanchor以外のパラメータも統合
        cfg = OmegaConf.merge(cfg, dotlist)

    if mode == "train":
        if cfg.dataset.fold is None:
            cfg.logger.runName = (
                f"{cfg.logger.runName}_all_data_seed_{cfg.seed}"
            )
        else:
            cfg.logger.runName = (
                f"{cfg.logger.runName}_fold_{cfg.dataset.fold}"
            )

    return cfg
