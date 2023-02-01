import snntorch as snn
import torch
from torch import nn


class MxSNNBlock(nn.Module):
    def __init__(self,
                 in_chn: int,
                 out_chn: int,
                 ksize: int,
                 lif_beta: float,
                 step: int = 1):
        """ A single SNN Block to be used in M-Models

        Args:
            in_chn: Ingoing Channels for Conv1d
            out_chn: Outgoing Channels for Conv1d
            ksize: Kernel Size for Conv1d
            lif_beta: Beta for Leaky & Fire
            step: Step Size for Conv1d
        """
        super().__init__()

        self.conv = nn.Conv1d(in_chn, out_chn, ksize, step, padding=int(ksize // 2 - 1))
        self.bn = nn.BatchNorm1d(out_chn)
        self.lif = snn.Leaky(beta=lif_beta)
        self.max_pool = nn.MaxPool1d(4)

    def init_leaky(self):
        return self.lif.init_leaky()

    def forward(self, x, mem: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x, mem = self.lif(x, mem)
        x = self.max_pool(x)
        return x, mem
