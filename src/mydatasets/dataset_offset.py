import ast
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class HMSOffsetDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        p_eeg_spec_root: str,
        specs,
        height: int,
        width: int,
        resize_h: int,
        resize_w: int,
        n_channel: int,
        use_channel: list,
        use_only_kaggle_specs: bool,
        use_only_eeg_specs: bool,
        order: str,
        group: str,
        label_smoothing_ver: str,
        label_smoothing_k: int,
        label_smoothing_epsilon: float,
        label_smoothing_n_evaluator: int,
        pseudo_label_n_evaluator: int,
        mode: str,
        fold: int,
        k_fold: int,
        spec_transforms=None,
        audio_transforms=None,
        p_fill_zero_some_channel: float = 0.0,
        p_swap_some_channel: float = 0.0,
        p_shift_time: float = 0.0,
        fill_zero_max_size: int = 1,
        swap_version: str = "ver_3",
        shift_max_ratio: int = 8,
        resize_kaggle_spec: bool = True,
        dry_run: int = None,
    ):
        self.csv_path = csv_path
        self.p_eeg_spec_root = p_eeg_spec_root
        self.specs = specs
        self.height = height
        self.width = width
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.n_channel = n_channel
        if order == "tile":
            assert self.n_channel % 4 == 0
            self.w_grid = self.n_channel // 4
        self.use_channel = use_channel
        self.use_only_kaggle_specs = use_only_kaggle_specs
        self.use_only_eeg_specs = use_only_eeg_specs
        self.order = order
        self.group = group
        self.label_smoothing_ver = label_smoothing_ver
        self.label_smoothing_k = label_smoothing_k
        self.label_smoothing_epsilon = label_smoothing_epsilon
        self.label_smoothing_n_evaluator = label_smoothing_n_evaluator
        self.pseudo_label_n_evaluator = pseudo_label_n_evaluator
        self.mode = mode
        self.spec_transforms = spec_transforms
        self.audio_transforms = audio_transforms
        self.p_fill_zero_some_channel = p_fill_zero_some_channel
        self.p_swap_some_channel = p_swap_some_channel
        self.p_shift_time = p_shift_time
        self.fill_zero_max_size = fill_zero_max_size
        self.swap_version = swap_version
        self.shift_max_ratio = shift_max_ratio
        self.resize_kaggle_spec = resize_kaggle_spec
        self.sr_eeg = 200  # [Hz]
        self.sr_spec = 0.5  # [Hz]
        self.dry_run = dry_run
        self.target_columns = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]

        df = pd.read_csv(self.csv_path)
        df = self.make_fold(df, fold, k_fold, mode)

        df["p_eeg_spec_root"] = p_eeg_spec_root

        df = self.build_csv(df)
        self.df = df
        if dry_run is not None:
            self.df = self.df[:dry_run]

        self.unique_eeg_id = self.df.eeg_id.unique()
        print(
            f"{self.mode} take unique: {len(self.df)} -> {len(self.unique_eeg_id)}"
        )

    def __len__(self):
        if self.mode not in ["test", "pseudo_label"]:
            return len(self.unique_eeg_id)
        else:
            return len(self.df)

    def __getitem__(self, index):
        self.index = index
        row = self._get_row(index)
        n_evaluator = row.n_evaluator_in_row
        spectrogram_id = row.spectrogram_id
        spectrogram_offset = row.spectrogram_label_offset_seconds
        label_id = row.label_id
        label = row[self.target_columns].values.astype("float32")
        p_eeg_spec_root = Path(row.p_eeg_spec_root)
        n_evaluator = row.n_evaluator_in_row

        if self.mode == "test":
            r = 0
        else:
            r = int(spectrogram_offset * self.sr_spec)

        data = np.zeros(
            (self.height, self.width, self.n_channel), dtype="float32"
        )
        for spec_channel in range(4):
            # img.shape = (100, 300)
            img = self.specs[spectrogram_id][
                r : r + 300, spec_channel * 100 : (spec_channel + 1) * 100
            ].T
            img = self.normalize9(img)

            if self.resize_kaggle_spec:
                img = cv2.resize(
                    img,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                data[:, :, spec_channel] = img
            else:
                # CROP TO 256 TIME STEPS
                data[14:-14, :, spec_channel] = img[:, 22:-22] / 2.0

        # EEG SPECTROGRAMS
        eeg_specs = np.load(
            p_eeg_spec_root / f"{label_id}.npy",
            allow_pickle=True,
        ).item()
        cnt = 0
        for k, img in eeg_specs.items():
            img = self.normalize9(img)
            if self.use_channel is None:
                data[:, :, cnt + 4] = img
                cnt += 1
            else:
                if k in self.use_channel:
                    data[:, :, cnt + 4] = img
                    cnt += 1

        # self.dump_data(data, prefix="before")
        if self.spec_transforms is not None:
            data = self.spec_transforms(image=data)["image"]
        # self.dump_data(data, prefix="after", stop_index=7)

        # self.dump_data(data, prefix="before_custom_aug")
        if self.mode == "train":
            data = self.custom_augmentation(data)
        # self.dump_data(data, prefix="after_custom_aug", stop_index=7)

        if self.order == "tile":
            # 4 x w_gridマスに(height, width)を並べて
            # (4 x height, w_grid x width)とする
            # (height, width) x 4 -> (4 x height, width)
            x_specs = [data[:, :, i] for i in range(4)]
            x_specs = np.concatenate(x_specs, axis=0)

            # (height, width) x N -> (w_grid - 1, 4, height, width)
            x_eeg_specs = (
                data[:, :, 4:]
                .transpose(2, 0, 1)
                .reshape(self.w_grid - 1, 4, self.height, self.width)
            )
            # -> (4, height, (w_grid - 1) x width)
            x_eeg_specs = np.concatenate(
                [x_eeg_specs[i] for i in range(self.w_grid - 1)], axis=-1
            )
            # -> (4 x height, (w_grid - 1) x width)
            x_eeg_specs = np.concatenate(
                [x_eeg_specs[i] for i in range(4)], axis=0
            )

            if self.use_only_kaggle_specs:
                # (4 x height, width)
                x = x_specs
            elif self.use_only_eeg_specs:
                # (4 x height, (w_grid - 1) x width)
                x = x_eeg_specs
            else:
                # -> (4 x height, w_grid x width)
                x = np.concatenate([x_eeg_specs, x_specs], axis=1)

            h, w = x.shape
            if self.resize_h != h or self.resize_w != w:
                x = cv2.resize(
                    x, (self.resize_w, self.resize_h), cv2.INTER_LINEAR
                )
            # -> (3, 4 x height, w_grid x width)
            data = np.stack([x, x, x], axis=0)
        elif self.order == "channel":
            # (h, w, c) -> (c, h, w)
            x_specs = data[:, :, :4].transpose(2, 0, 1)
            x_eeg_specs = data[:, :, 4:].transpose(2, 0, 1)

            if self.use_only_kaggle_specs:
                data = x_specs
            elif self.use_only_eeg_specs:
                data = x_eeg_specs
            else:
                data = np.concatenate([x_eeg_specs, x_specs], axis=0)
        if self.mode == "train":
            label = self.label_smoothing(label, n_evaluator)
        return data, label, n_evaluator

    def _get_row(self, index):
        if self.mode not in ["test", "pseudo_label"]:
            if self.mode == "train":
                try:
                    return (
                        self.df[self.df["eeg_id"] == self.unique_eeg_id[index]]
                        .sample(n=1)
                        .iloc[0]
                    )
                except ValueError as e:
                    print("error: eeg_id = ", self.unique_eeg_id[index])
                    print(e)

            else:
                tmp = self.df[
                    self.df["eeg_id"] == self.unique_eeg_id[index]
                ].reset_index(drop=True)
                choise = len(tmp) // 2
                return tmp.iloc[choise]
                # return tmp.iloc[0]

        else:
            return self.df.iloc[index]

    def normalize9(self, img):
        ep = 1e-6
        img = np.log(img + ep)
        img = np.nan_to_num(img, nan=0.0)
        return img

    def custom_augmentation(self, data):
        h, w, c = data.shape
        if np.random.rand() < self.p_fill_zero_some_channel:
            # いくつかのチャンネルをランダムに0にする
            size = np.random.randint(self.fill_zero_max_size)
            channels = np.random.choice(np.arange(c), size=size, replace=False)
            data[..., channels] = 0
        if np.random.rand() < self.p_swap_some_channel:
            # チャンネルの並び順をランダムに変更する
            if self.swap_version == "ver_5":
                # Left,Rightの塊で入れ替える(24chバージョン)
                assert data.shape[-1] == 24
                # fmt: off
                swap_index = [
                    2, 3, 0, 1, 6, 7, 4, 5, 
                    16, 17, 18, 19, 20, 21, 22, 23, 
                    8, 9, 10, 11, 12, 13, 14, 15
                ]
                # fmt: on
                data = data[..., swap_index]
        if np.random.rand() < self.p_shift_time:
            # 時間軸方向にランダムにshift
            max_shift = w // self.shift_max_ratio
            shift = np.random.randint(-max_shift, max_shift)
            data = np.roll(data, shift, axis=1)
        return data

    def do_custom_label_smoothing(self, label, epsilon):
        # ピークが複数ある場合、全てのピークを落としてそれ以外を持ち上げる
        max_value = np.max(label)
        idx_max = np.where(label == max_value)[0]
        is_max = np.identity(6)[idx_max].sum(axis=0).astype("bool")
        n_idx_max = is_max.sum()
        new_epsilon = epsilon / n_idx_max
        label[is_max] -= new_epsilon
        label[~is_max] += epsilon / (6 - n_idx_max)
        return label

    def label_smoothing(self, label, n_evaluator):
        if self.label_smoothing_ver == "ver_1":
            epsilon = 1 / (self.label_smoothing_k + np.sqrt(n_evaluator))
            label = self.do_custom_label_smoothing(label, epsilon)
        elif self.label_smoothing_ver == "ver_2":
            if n_evaluator <= self.label_smoothing_n_evaluator:
                label = self.do_custom_label_smoothing(
                    label, self.label_smoothing_epsilon
                )
        return label

    def build_csv(self, df):
        print("#" * 20)
        if self.group is not None:
            if self.mode in ["train", "pseudo_label"]:
                org_shape = df.shape
                if self.group == "low_vote":
                    df = df[df["n_evaluator_in_row"] < 10]
                elif self.group == "high_vote":
                    df = df[df["n_evaluator_in_row"] >= 10]
                print(
                    f"{self.mode} {self.group} group: {org_shape} -> {df.shape}"
                )
            else:
                print(f"{self.mode}: {df.shape}")
        return df

    @classmethod
    def _make_fold(cls, df, fold, k_fold, mode):
        df_new = df.copy()

        if -1 in df_new.fold.tolist():
            if fold == -1:
                df_new = df_new[df_new.fold == -1]
            else:
                df_new = df_new[df_new.fold != -1]

        if fold != -1:
            n_fold = df_new.fold.nunique()
            offset = n_fold // k_fold
            target = [i + fold * offset for i in range(offset)]

            if mode == "train":
                df_new = df_new.query(f"fold not in {target}")
            else:
                df_new = df_new.query(f"fold in {target}")
        else:
            df_new = df_new.query(f"fold == -1")

        return df_new

    @classmethod
    def make_fold(cls, df, fold, k_fold, mode):
        if fold is not None:
            df = cls._make_fold(df, fold, k_fold, mode)
        else:
            if mode in ["train", "test", "pseudo_label"]:
                df = df  # all data
            elif mode == "val":
                df = df[:1]  # dummy data
            else:
                raise NotImplementedError
        df = df.reset_index(drop=True)
        return df

    def dump_data(self, image, label=None, prefix=None, stop_index=None):
        p_output_root = Path("./tmp/aug_albu/")
        p_output_root.mkdir(exist_ok=True, parents=True)
        filename = f"{self.mode}_{self.index}.npz"
        if prefix is not None:
            filename = f"{prefix}_" + filename
        np.savez(p_output_root / filename, image=image, label=label)
        if self.index == stop_index:
            exit()
