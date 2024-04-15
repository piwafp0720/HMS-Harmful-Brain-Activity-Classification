import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from timm import models

from .layers import pooling


class HMS2DModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        pretrained: bool = True,
        pool_type: str = "avg",
        n_hiddens: int = 512,
        n_classes: int = 2,
        drop_path_rate: float = 0.0,
        drop_rate_backbone: float = 0.0,
        drop_rate_fc: float = 0.0,
        manifold_mixup_alpha: float = 0.4,
    ):
        super().__init__()

        pretrained_cfg = None
        if "." in backbone:
            backbone, pretrained_cfg = backbone.split(".")

        self.backbone = getattr(models, backbone)(
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate_backbone,
            drop_path_rate=drop_path_rate,
            pretrained_cfg=pretrained_cfg,
        )

        if pool_type == "avg":
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pooling = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "gem":
            self.pooling = pooling.GeM()
        elif pool_type == "mac":
            self.pooling = pooling.MAC()
        else:
            raise KeyError

        self.feat_1 = nn.LazyLinear(n_hiddens)
        self.feat_2 = nn.LazyLinear(n_classes)
        self.dropout = nn.Dropout(drop_rate_fc)
        self.bn = nn.BatchNorm1d(n_hiddens)
        self.mixup_alpha = manifold_mixup_alpha

    def mixup(self, features):
        lam = (
            np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if self.mixup_alpha > 0
            else 1
        )

        index = torch.randperm(features.size()[0]).type_as(features).long()

        features = lam * features + (1 - lam) * features[index]
        return features, lam, index

    def forward_until_pooling(self, x):
        x = self.backbone.forward_features(x)
        if x.dim() == 4:  # CNN family
            x = self.pooling(x).squeeze(-1).squeeze(-1)
        if x.dim() == 3:  # ViT family
            if self.backbone.global_pool == "avg":
                # TODO: The first token is excluded in some models.
                x = x.mean(dim=1)
            else:
                x = x[:, 0, :]
            if (
                hasattr(self.backbone, "fc_norm")
                and self.backbone.fc_norm is not None
            ):
                x = self.backbone.fc_norm(x)

        return x

    def forward(self, x, manifold_mixup=False):
        x = self.forward_until_pooling(x)

        if manifold_mixup:
            x, lam, index = self.mixup(x)

        x = self.feat_2(torch.relu(self.bn(self.feat_1(self.dropout(x)))))

        if manifold_mixup:
            return x, lam, index
        else:
            return x


class HMS2DModelV3(nn.Module):
    def __init__(
        self,
        backbone: str,
        neck: str,
        in_chans: int,
        ver: str = "ver_1",
        pretrained: bool = True,
        pool_type: str = "avg",
        n_hiddens: int = 512,
        n_classes: int = 2,
        drop_path_rate: float = 0.0,
        drop_rate_backbone: float = 0.0,
        drop_rate_fc: float = 0.0,
        manifold_mixup_alpha: float = 0.4,
    ):
        super().__init__()

        self.backbone = self.get_timm_model(
            backbone,
            pretrained,
            in_chans,
            drop_rate_backbone,
            drop_path_rate,
        )

        if pool_type == "avg":
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pooling = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "gem":
            self.pooling = pooling.GeM()
        elif pool_type == "mac":
            self.pooling = pooling.MAC()
        else:
            raise KeyError

        self.mixup_alpha = manifold_mixup_alpha

        self.ver = ver
        if self.ver == "ver_4":
            # backboneの途中のfeature mapに対してchannel方向にpoolingした画像を
            # タイル状に並べるバージョン
            self.backbone = self.get_timm_model(
                backbone,
                pretrained,
                in_chans,
                drop_rate_backbone,
                drop_path_rate,
                features_only=True,
            )
            self.neck = self.get_timm_model(neck)
            self.feat_1 = nn.LazyLinear(n_hiddens)
            self.feat_2 = nn.LazyLinear(n_classes)
            self.dropout = nn.Dropout(drop_rate_fc)
            self.bn = nn.BatchNorm1d(n_hiddens)
        elif self.ver == "ver_7":
            # backboneの途中のfeature mapを全部並べるバージョン
            self.backbone = self.get_timm_model(
                backbone,
                pretrained,
                in_chans,
                drop_rate_backbone,
                drop_path_rate,
                features_only=True,
            )
            self.neck = self.get_timm_model(neck)
            self.feat_1 = nn.LazyLinear(n_hiddens)
            self.feat_2 = nn.LazyLinear(n_classes)
            self.dropout = nn.Dropout(drop_rate_fc)
            self.bn = nn.BatchNorm1d(n_hiddens)

    def get_timm_model(
        self,
        backbone,
        pretrained=True,
        in_chans=1,
        drop_rate_backbone=0.0,
        drop_path_rate=0.0,
        features_only=False,
    ):
        pretrained_cfg = None
        if "." in backbone:
            backbone, pretrained_cfg = backbone.split(".")

        model = getattr(models, backbone)(
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate_backbone,
            drop_path_rate=drop_path_rate,
            pretrained_cfg=pretrained_cfg,
            features_only=features_only,
        )
        return model

    def mixup(self, features):
        lam = (
            np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if self.mixup_alpha > 0
            else 1
        )

        index = torch.randperm(features.size()[0]).type_as(features).long()

        features = lam * features + (1 - lam) * features[index]
        return features, lam, index

    def forward_until_pooling(self, model, x, with_pooling):
        x = model.forward_features(x)
        if x.dim() == 4:  # CNN family
            if with_pooling:
                x = self.pooling(x).squeeze(-1).squeeze(-1)
        if x.dim() == 3:  # ViT family
            if with_pooling:
                if model.global_pool == "avg":
                    # TODO: The first token is excluded in some models.
                    x = x.mean(dim=1)
                else:
                    x = x[:, 0, :]
                if hasattr(model, "fc_norm") and model.fc_norm is not None:
                    x = model.fc_norm(x)
            else:
                x = x.unsqueeze(1)

        return x

    def forward(self, x, manifold_mixup=False):
        if x.dim() == 4:
            # (b, c, h, w) -> (b * c, 1, h, w)
            b, c, h, w = x.shape
            assert c > 3
            x = x.reshape(-1, 1, h, w)
        elif x.dim() == 5:
            # (b, c, 2, h, w) -> (b * c, 2, h, w)
            b, c, _, h, w = x.shape
            assert c > 3
            x = x.reshape(-1, 2, h, w)

        if self.ver == "ver_4":
            # backboneの途中のfeature mapに対してchannel方向にpoolingした画像を
            # タイル状に並べるバージョン
            # (b * c,  f_backbone, mini_h, mini_w)
            x = self.backbone(x)[4]
            # (b * c, 1, mini_h, mini_w)
            x = torch.mean(x, dim=1, keepdims=True)
            _, _, mini_h, mini_w = x.shape
            # (b, c, mini_h, mini_w)
            x = x.reshape(b, c, mini_h, mini_w)
            if manifold_mixup:
                x, lam, index = self.mixup(x)
            x = self.tiling(x)
            # (b, 1, 256, 256)
            x = F.interpolate(x, (256, 256), mode="nearest")
            # (b, f_neck)
            x = self.forward_until_pooling(self.neck, x, True)
            # (b, n_classes)
            x = self.feat_2(torch.relu(self.bn(self.feat_1(self.dropout(x)))))
        elif self.ver == "ver_7":
            # backboneの途中のfeature mapを全部並べるバージョン
            # (b * c,  f_backbone, mini_h, mini_w)
            x = self.backbone(x)[4]
            _, n_features, mini_h, mini_w = x.shape
            x = x.reshape(b, c, n_features, mini_h, mini_w)
            # (b, n_features, c, mini_h, mini_w)
            x = x.permute(0, 2, 1, 3, 4)
            # (b, 1, n_features, c x mini_h x mini_w)
            x = x.reshape(b, 1, n_features, -1)
            # (b, 1, 256, 256)
            x = F.interpolate(x, (256, 256), mode="nearest")
            if manifold_mixup:
                x, lam, index = self.mixup(x)
            # (b, f_neck)
            x = self.forward_until_pooling(self.neck, x, True)
            # (b, n_classes)
            x = self.feat_2(torch.relu(self.bn(self.feat_1(self.dropout(x)))))

        if manifold_mixup:
            return x, lam, index
        else:
            return x

    def tiling(self, x):
        b, c, h, w = x.shape
        assert c % 4 == 0
        n_rows = 4
        n_cols = c // 4
        # 2つの列を表すリストを準備
        columns = []
        # 列優先で画像をリストに追加
        for j in range(n_cols):
            column_images = []
            for i in range(n_rows):
                # 対応する画像を列のリストに追加
                column_images.append(x[:, i + j * n_rows, :, :])
            # 一つの列に属する画像を垂直方向（dim=1）に結合
            columns.append(torch.cat(column_images, dim=1))

        # 最終的に列を水平方向（dim=2）に結合して、全体の画像を形成
        x = torch.cat(columns, dim=2)
        # (b, 1, new_h, new_w)
        x = x.unsqueeze(1)
        return x
