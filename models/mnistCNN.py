import torch
import torch.nn as nn

__all__ = [
    "CNNmnist",
]


class CNNmnist(nn.Module):
    def __init__(self, classes=10):
        super(CNNmnist, self).__init__()
        self.act = nn.ReLU()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(3136, 100)
        self.fc2 = nn.Linear(100, classes)

    def forward(self, x):
        x = self.body(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x)
