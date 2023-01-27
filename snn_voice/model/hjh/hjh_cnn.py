from abc import ABC
from collections import OrderedDict

import torch
from torch import nn

from snn_voice.model.module import ModuleCNN
from snn_voice.model.hjh.blocks import HjhCNNBlock


class HjhCNN(ModuleCNN, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_blks = nn.Sequential(
            OrderedDict([
                ('conv_blk1', HjhCNNBlock(1, 8, 5, 2)),
                ('conv_blk2', HjhCNNBlock(8, 16, 5)),
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Sequential(nn.Linear(16, self.n_classes))),
            ])
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.example_input_array = torch.rand([32, 1, 60, 101])

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x).squeeze()
        x = self.classifier(x)

        return x
