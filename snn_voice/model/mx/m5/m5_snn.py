from collections import OrderedDict

from torch import nn

from snn_voice.model.mx.blocks import MxSNNBlock


def m5_snn_init(self, lif_beta: float, n_classes: int):
    """ Initializes the M5 __init__ blocks """
    self.net = nn.Sequential(
        OrderedDict([
            ('snn1', MxSNNBlock(1, 128, 80, lif_beta, 4)),
            ('snn2', MxSNNBlock(128, 128, 3, lif_beta)),
            ('snn3', MxSNNBlock(128, 256, 3, lif_beta)),
            ('snn4', MxSNNBlock(256, 512, 3, lif_beta)),

            ('avg_pool', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten(start_dim=1)),

            ('fc', nn.Linear(512, n_classes))
        ])
    )
