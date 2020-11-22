from torch import nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        height = x.data.size(2)
        width = x.data.size(3)
        x = F.avg_pool2d(x, (height, width))
        x = x.view(N, C)
        return x