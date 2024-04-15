from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HMSLoss(torch.nn.Module):
    def __init__(self, ver):
        super(HMSLoss, self).__init__()
        if ver == "ver_1":
            self.loss_fn_list = [nn.KLDivLoss(reduction="batchmean")]
            loss_weight = [1]

        self.register_buffer(
            "loss_weight",
            torch.tensor(
                loss_weight,
                dtype=torch.float,
            ),
        )

    def forward(self, y_pred, y_true):
        loss_dict = {}
        loss_list = []
        for loss_fn, weight in zip(self.loss_fn_list, self.loss_weight):
            loss_name = type(loss_fn).__name__
            if loss_name == "KLDivLoss":
                loss = loss_fn(F.log_softmax(y_pred, dim=1), y_true) * weight
            else:
                loss = loss_fn(y_pred, y_true) * weight
            loss_dict[loss_name] = loss
            loss_list.append(loss)
        loss = torch.stack(loss_list).sum()
        loss_dict["loss"] = loss
        return loss_dict, loss


class HMSSEDLoss(torch.nn.Module):
    def __init__(self, ver):
        super(HMSSEDLoss, self).__init__()
        if ver == "ver_1":
            self.loss_fn_clipwise_list = [nn.KLDivLoss(reduction="batchmean")]
            self.loss_fn_segmentwise_list = [
                nn.KLDivLoss(reduction="batchmean")
            ]
            loss_weight_clipwise = [1]
            loss_weight_segmentwise = [1]
        elif ver == "ver_2":
            self.loss_fn_clipwise_list = [nn.KLDivLoss(reduction="batchmean")]
            self.loss_fn_segmentwise_list = [
                nn.KLDivLoss(reduction="batchmean")
            ]
            loss_weight_clipwise = [1]
            loss_weight_segmentwise = [5]
        elif ver == "ver_3":
            self.loss_fn_clipwise_list = [
                nn.BCEWithLogitsLoss(reduction="mean")
            ]
            self.loss_fn_segmentwise_list = [
                nn.BCEWithLogitsLoss(reduction="none")
            ]
            loss_weight_clipwise = [1]
            loss_weight_segmentwise = [1]
        elif ver == "ver_4":
            self.loss_fn_clipwise_list = [
                nn.KLDivLoss(reduction="batchmean"),
                nn.BCEWithLogitsLoss(reduction="mean"),
            ]
            self.loss_fn_segmentwise_list = [
                nn.KLDivLoss(reduction="batchmean"),
                nn.BCEWithLogitsLoss(reduction="none"),
            ]
            loss_weight_clipwise = [1, 1]
            loss_weight_segmentwise = [1, 1]
        elif ver == "ver_5":
            self.loss_fn_clipwise_list = [
                nn.KLDivLoss(reduction="batchmean"),
                nn.BCEWithLogitsLoss(reduction="mean"),
            ]
            self.loss_fn_segmentwise_list = [
                nn.KLDivLoss(reduction="batchmean"),
                nn.BCEWithLogitsLoss(reduction="none"),
            ]
            loss_weight_clipwise = [1, 5]
            loss_weight_segmentwise = [3, 3]

        self.register_buffer(
            "loss_weight_clipwise",
            torch.tensor(
                loss_weight_clipwise,
                dtype=torch.float,
            ),
        )
        self.register_buffer(
            "loss_weight_segmentwise",
            torch.tensor(
                loss_weight_segmentwise,
                dtype=torch.float,
            ),
        )

    def forward(self, clipwise_output, segmentwise_output, y_true):
        loss_dict = {}
        loss_list = []

        # loss for clipwise_output
        for loss_fn, weight in zip(
            self.loss_fn_clipwise_list, self.loss_weight_clipwise
        ):
            loss_name = type(loss_fn).__name__
            if loss_name == "KLDivLoss":
                loss = (
                    loss_fn(F.log_softmax(clipwise_output, dim=1), y_true)
                    * weight
                )
            else:
                loss = loss_fn(clipwise_output, y_true) * weight
            loss_dict[f"{loss_name}_clipwise"] = loss
            loss_list.append(loss)

        # loss for segmentwise_output
        b, h, w = segmentwise_output.shape
        # (b, n_class) -> (b, n_class, 1) -> (b, n_class, w)
        y_true = y_true.unsqueeze(-1).expand(-1, -1, w)
        mask = torch.zeros_like(y_true).bool()
        middle = w / 2
        window_10s = 10 / 50 * w
        start = int(middle - window_10s / 2)
        end = int(start + window_10s) + 1
        mask[:, :, start:end] = 1
        for loss_fn, weight in zip(
            self.loss_fn_segmentwise_list, self.loss_weight_segmentwise
        ):
            loss_name = type(loss_fn).__name__
            if loss_name == "KLDivLoss":
                loss = (
                    loss_fn(
                        F.log_softmax(segmentwise_output, dim=1)[mask],
                        y_true[mask],
                    )
                    * weight
                )
            else:
                loss = loss_fn(segmentwise_output, y_true) * weight
                loss = loss[mask].mean()
            loss_dict[f"{loss_name}_segmentwise"] = loss
            loss_list.append(loss)

        loss = torch.stack(loss_list).sum()
        loss_dict["loss"] = loss
        return loss_dict, loss
