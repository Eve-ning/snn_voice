import snntorch as snn
from torch import nn


class PiczakSNNBlock(nn.Module):
    def __init__(
            self,
            in_chn, out_chn,
            ksize, step,
            max_pool_ksize, max_pool_step,
            lif_beta: float, dropout=0.5
    ):
        super().__init__()

        conv = nn.Conv2d(in_chn, out_chn, ksize, step)
        max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_step)
        dropout = nn.Dropout(dropout)
        lif = snn.Leaky(beta=lif_beta, init_hidden=True)
        self.net = nn.Sequential(conv, max_pool, dropout, lif)

    def forward(self, x):
        return self.net(x)
