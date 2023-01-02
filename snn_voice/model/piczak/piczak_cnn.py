from collections import OrderedDict

from torch import nn

from snn_voice.model.mx.mx_cnn import MxCNN
from snn_voice.model.piczak.piczak_cnn_block import PiczakCNNBlock


class PiczakCNN(MxCNN):

    def __init__(self, n_channel: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_blks = nn.Sequential(
            OrderedDict([
                ('conv_blk1', PiczakCNNBlock(1, n_channel, 57, 6, 1, 1, 4, 3, 1, 3)),
                ('conv_blk2', PiczakCNNBlock(n_channel, n_channel, 1, 3, 1, 1, 1, 3, 1, 3)),
            ])
        )

        self.classifier = nn.Linear(n_channel, self.n_classes)
