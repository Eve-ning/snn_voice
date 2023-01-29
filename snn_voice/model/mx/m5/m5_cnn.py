from collections import OrderedDict

from torch import nn

from snn_voice.model.module import ModuleCNN
from snn_voice.model.mx.blocks import MxCNNBlock


class M5CNN(ModuleCNN, nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            OrderedDict([
                ('cnn',
                 nn.Sequential(MxCNNBlock(1, 128, 80, 4),
                               MxCNNBlock(128, 128, 3, 1),
                               MxCNNBlock(128, 256, 3, 1),
                               MxCNNBlock(256, 512, 3, 1))
                 ),
                ('avg_pool', nn.AdaptiveAvgPool1d(1)),
                ('flatten', nn.Flatten(start_dim=1)),
                ('fc', nn.Linear(512, self.n_classes))
            ])
        )

    def forward(self, x):
        return self.net(x)
