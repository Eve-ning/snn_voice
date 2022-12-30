from collections import OrderedDict
from dataclasses import dataclass, field

import torch.nn as nn

from src.model.mx.m5_block import M5Block
from src.model.mx.m5_common import M5Common


@dataclass
class M5(M5Common):
    conv_blks: nn.Module = field(
        init=False,
        default_factory=lambda: nn.Sequential(
            OrderedDict([
                ('conv_blk1', M5Block(1, 128, 80, 4)),
                ('conv_blk2', M5Block(128, 128, 3, 1)),
                ('conv_blk3', M5Block(128, 256, 3, 1)),
                ('conv_blk4', M5Block(256, 512, 3, 1)),
            ])
        )
    )

    def __post_init__(self):
        super().__post_init__()
