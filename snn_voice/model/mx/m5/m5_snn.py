from collections import OrderedDict

from torch import nn

from snn_voice.model.mx.blocks import MxSNNBlock


class M5SNN(nn.Module):
    def __init__(self, lif_beta: float, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ('snn',
                 nn.Sequential(
                     MxSNNBlock(1, 128, 80, lif_beta, 4),
                     MxSNNBlock(128, 128, 3, lif_beta),
                     MxSNNBlock(128, 256, 3, lif_beta),
                     MxSNNBlock(256, 512, 3, lif_beta),
                 )),

                ('avg_pool', nn.AdaptiveAvgPool1d(1)),
                ('flatten', nn.Flatten(start_dim=1)),

                ('fc', nn.Linear(512, n_classes))
            ])
        )

    def forward(self, x):
        return self.net(x)
