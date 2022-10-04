import torch
import torch.nn as nn


def SetOptimizer(model:nn.Module,lr:float,**kwargs) -> torch.optim:

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,**kwargs)

    return optimizer
