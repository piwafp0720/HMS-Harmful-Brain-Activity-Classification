"""
https://github.com/hongyi-zhang/mixup/blob/master/cifar/easy_mixup.py
"""
import numpy as np
import torch


def mixup_data(x, y, alpha=0.4):
    """
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the shuffle batch
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
