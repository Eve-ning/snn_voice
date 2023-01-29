from abc import ABC
from collections import OrderedDict

from torch import nn

from snn_voice.model.module import ModuleCNN
from snn_voice.model.piczak.blocks import PiczakCNNBlock


class PiczakCNN(ModuleCNN, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net = nn.Sequential(
            OrderedDict([
                ('cnn',
                 nn.Sequential(
                     PiczakCNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), 0.5),
                     PiczakCNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), None))
                 ),

                ('avg_pool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', nn.Flatten(start_dim=1)),

                ('fc',
                 nn.Sequential(
                     nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU()),
                     nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU()),
                     nn.Sequential(nn.Linear(5000, self.n_classes))
                 ),
                 )
            ])
        )

    def forward(self, x):
        return self.net(x)
