from typing import Tuple

import snntorch as snn
import torch
from torch import nn


class PiczakSNNBlock(nn.Module):
    def __init__(
            self,
            in_chn: int, out_chn: int,
            ksize: Tuple[int, int],
            step: Tuple[int, int],
            max_pool_ksize: Tuple[int, int],
            max_pool_step: Tuple[int, int],
            beta: float,
            dropout: float = 0.5,
            learn_beta: bool = True,
            learn_thres: bool = False,
    ):
        """ A single SNN Block to be used in Piczak Models

        Args:
            in_chn: Ingoing Channels for Conv2d
            out_chn: Outgoing Channels for Conv2d
            ksize: Kernel Size for Conv2d
            step: Step Size for Conv2d
            max_pool_ksize: Max Pool 2D Kernel Size
            max_pool_step: Max Pool 2D Step Size
            beta: Beta for Leaky & Fire
            dropout: Dropout probability
            learn_beta: Whether to set beta to be learnable
            learn_thres: Whether to set threshold to be learnable
        """
        super().__init__()

        self.conv = nn.Conv2d(in_chn, out_chn, ksize, step)
        self.max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_step)
        self.dropout = nn.Dropout(dropout)
        self.lif = snn.Leaky(beta=beta, learn_beta=learn_beta, learn_threshold=learn_thres)

    def forward(self, x, mem: torch.Tensor):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x, mem = self.lif(x, mem)
        return x, mem

    def init_leaky(self):
        return self.lif.init_leaky()
