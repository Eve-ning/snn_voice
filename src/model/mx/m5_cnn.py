from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn

from src.model.mx.mx_cnn import MxCNN
from src.model.mx.mx_cnn_block import MxCNNBlock


class M5CNN(MxCNN, nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_blks = nn.Sequential(
            OrderedDict([
                ('conv_blk1', MxCNNBlock(1, 128, 80, 4)),
                ('conv_blk2', MxCNNBlock(128, 128, 3, 1)),
                ('conv_blk3', MxCNNBlock(128, 256, 3, 1)),
                ('conv_blk4', MxCNNBlock(256, 512, 3, 1)),
            ])
        )

        self.classifier = nn.Linear(512, self.n_classes)
