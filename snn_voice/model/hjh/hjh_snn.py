from collections import OrderedDict

import torch
from torch import nn

from snn_voice.model.hjh.blocks import HjhCNNBlock


def hjh_snn_init(self):
    """ Initializes the Hjh __init__ blocks """
    self.conv_blks = nn.Sequential(
        OrderedDict([
            # TODO: Insert the Leaky here
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
