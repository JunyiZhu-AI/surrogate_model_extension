import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, classes=10):
        super(MLP, self).__init__()
        # act = nn.LeakyReLU(negative_slope=1e-2)
        act = nn.ReLU()
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(784, 1000)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(1000, 1000)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(1000, classes)),
                ('act', act)
            ]))
        ])

    def forward(self, x):
        for layer in self.body:
            if isinstance(layer.layer, nn.Linear):
                x = x.flatten(1)
            x = layer(x)
        return x
