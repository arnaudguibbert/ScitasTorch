import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self,**kwargs) -> None:
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(1,8,5),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(8,16,3),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(16),
                                  nn.Conv2d(16,32,3),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(32),
                                  nn.MaxPool2d(2),
                                  )
        self.FC = nn.Sequential(nn.Linear(4*4*32,64),
                                nn.ReLU(),
                                nn.Linear(64,10))

    def forward(self,x:torch.tensor) -> torch.tensor:
        feat = self.conv(x)
        out = self.FC(feat.reshape(x.shape[0],-1))
        return out
