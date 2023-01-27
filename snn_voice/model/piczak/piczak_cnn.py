from abc import ABC
from collections import OrderedDict

import torch
from torch import nn

from snn_voice.model.module import ModuleCNN
from snn_voice.model.piczak.blocks import PiczakCNNBlock


class PiczakCNN(ModuleCNN, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_blks = nn.Sequential(
            OrderedDict([
                ('conv_blk1', PiczakCNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), 0.5)),
                ('conv_blk2', PiczakCNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), None)),
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU())),
                ('fc2', nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU())),
                ('fc3', nn.Sequential(nn.Linear(5000, self.n_classes)))
            ])
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.example_input_array = torch.rand([32, 1, 60, 101])

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x).squeeze()
        x = self.classifier(x)

        return x
