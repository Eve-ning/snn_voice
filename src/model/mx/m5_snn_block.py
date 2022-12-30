import snntorch as snn
from torch import nn


class M5SNNBlock(nn.Module):
    def __init__(self, in_chn, out_chn, ksize,
                 lif_beta, step=1):
        super().__init__()

        self.conv = nn.Conv1d(in_chn, out_chn, ksize, step,
                              padding=int(ksize // 2 - 1))
        self.bn = nn.BatchNorm1d(out_chn)
        self.lif = snn.Leaky(beta=lif_beta)
        self.max_pool = nn.MaxPool1d(4)

    def init_leaky(self):
        return self.lif.init_leaky()

    def forward(self, x, mem):
        x = self.conv(x)
        x = self.bn(x)
        x, mem = self.lif(x, mem)
        x = self.max_pool(x)
        return x, mem
