from collections import OrderedDict

from torch import nn

from snn_voice.model.mx import MxSNNBlock


def m5_snn_init(self):
    """ Initializes the M5 __init__ blocks """
    self.conv_blks = nn.Sequential(
        OrderedDict([
            ('conv_blk1', MxSNNBlock(1, 128, 80, self.lif_beta, 4)),
            ('conv_blk2', MxSNNBlock(128, 128, 3, self.lif_beta)),
            ('conv_blk3', MxSNNBlock(128, 256, 3, self.lif_beta)),
            ('conv_blk4', MxSNNBlock(256, 512, 3, self.lif_beta)),
        ])
    )
    self.classifier = nn.Linear(512, self.n_classes)
