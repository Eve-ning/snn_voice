from collections import OrderedDict

from torch import nn

from snn_voice.model.module.module_snn_latency import ModuleSNNLatency
from snn_voice.model.mx.mx_snn_block import MxSNNBlock


class M5SNNLatency(ModuleSNNLatency):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_blks = nn.Sequential(
            OrderedDict([
                ('conv_blk1', MxSNNBlock(1, 128, 80, self.lif_beta, 4)),
                ('conv_blk2', MxSNNBlock(128, 128, 3, self.lif_beta)),
                ('conv_blk3', MxSNNBlock(128, 256, 3, self.lif_beta)),
                ('conv_blk4', MxSNNBlock(256, 512, 3, self.lif_beta)),
            ])
        )
        self.classifier = nn.Linear(512, self.n_classes)
