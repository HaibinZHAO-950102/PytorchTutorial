import torch

def acc(prediction, y):
    act, idx = torch.max(prediction, dim=1)
    corrects = (y.view(-1) == idx)
    return corrects.float().sum().item() / y.numel()

def ACC(nn, x, y, *args, **kwargs):
    act, idx = torch.max(nn(x), dim=1)
    corrects = (y.view(-1) == idx)
    return corrects.float().sum().item() / y.numel()