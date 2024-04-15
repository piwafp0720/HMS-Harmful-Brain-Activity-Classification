import numpy as np
import torch


def timemix_data(x, y, alpha=1.0):
    """
    Applies cutmix to a sample
    Arguments:
        x {torch tensor} -- Input batch (batchsize, c, h, w)
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the shuffle batch
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).type_as(x).long()

    y_a, y_b = y, y[index]
    bbx1, bbx2 = rand_bbox(x.size(), lam)
    x[:, ..., bbx1:bbx2] = x[index, ..., bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * x.size()[-2] / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    w = size[-1]
    cut_rat = 1.0 - lam
    cut_w = np.int(w * cut_rat)

    # uniform
    cx = np.random.randint(w)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)

    return bbx1, bbx2
