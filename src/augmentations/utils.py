def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_dict_a, loss_a = criterion(pred, y_a)
    loss_dict_b, loss_b = criterion(pred, y_b)
    loss = lam * loss_a + (1 - lam) * loss_b
    loss_dict = {}
    for k in loss_dict_a.keys():
        loss_dict[k] = lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k]
    return loss_dict, loss
