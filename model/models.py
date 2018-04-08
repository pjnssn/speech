import torch
from torch import nn


def make_block(in_channels, out_channels, kernel_size=5, padding=2, mp_kernel_size=4):
    block = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if mp_kernel_size:
        block.append(nn.MaxPool1d(kernel_size=mp_kernel_size))
    return nn.Sequential(*block)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block_1 = make_block(1, 8)
        self.block_2 = make_block(8, 16)
        self.block_3 = make_block(16, 32)
        self.block_4 = make_block(32, 64)
        self.block_5 = make_block(64, 128)
        self.block_6 = make_block(128, 256, mp_kernel_size=None)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 12),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = torch.squeeze(self.avg(x))
        x = self.classifier(x)
        return x

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
