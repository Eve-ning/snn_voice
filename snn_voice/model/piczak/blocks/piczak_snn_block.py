import snntorch as snn
from torch import nn


class PiczakSNNBlock(nn.Module):
    def __init__(
            self,
            in_chn: int, out_chn: int,
            ksize: int, step: int,
            max_pool_ksize: int, max_pool_step,
            lif_beta: float, dropout: float = 0.5
    ):
        """ A single SNN Block to be used in Piczak Models

        Args:
            in_chn: Ingoing Channels for Conv2d
            out_chn: Outgoing Channels for Conv2d
            ksize: Kernel Size for Conv2d
            step: Step Size for Conv2d
            max_pool_ksize: Max Pool 2D Kernel Size
            max_pool_step: Max Pool 2D Step Size
            lif_beta: Beta for Leaky & Fire
            dropout: Dropout probability
        """
        super().__init__()

        conv = nn.Conv2d(in_chn, out_chn, ksize, step)
        max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_step)
        dropout = nn.Dropout(dropout)
        lif = snn.Leaky(beta=lif_beta, init_hidden=True)
        self.net = nn.Sequential(conv, max_pool, dropout, lif)

    def forward(self, x):
        return self.net(x)
