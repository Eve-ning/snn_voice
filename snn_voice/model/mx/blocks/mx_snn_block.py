import snntorch as snn
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

        conv = nn.Conv1d(in_chn, out_chn, ksize, step,
                         padding=int(ksize // 2 - 1))
        bn = nn.BatchNorm1d(out_chn)
        lif = snn.Leaky(beta=lif_beta)
        max_pool = nn.MaxPool1d(4)

        self.net = nn.Sequential(conv, bn, lif, max_pool)

    def forward(self, x):
        return self.net(x)
