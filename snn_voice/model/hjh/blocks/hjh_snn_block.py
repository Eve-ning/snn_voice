import snntorch as snn
from torch import nn


class HjhSNNBlock(nn.Module):
    def __init__(self,
                 in_chn: int,
                 out_chn: int,
                 ksize: int,
                 beta: float,
                 max_pool_ksize: int = None,
                 learn_beta: bool = True,
                 learn_thres: bool = True,
                 ):
        """ A single SNN Block to be used in Hjh Models

        Args:
            in_chn: Ingoing Channels for Conv1d
            out_chn: Outgoing Channels for Conv1d
            ksize: Kernel Size for Conv1d
            beta: Beta for Leaky & Fire
            max_pool_ksize: Max Pool Kernel Size, step size will be the same
            learn_beta: Whether to set beta to be learnable
            learn_thres: Whether to set threshold to be learnable
        """
        super().__init__()

        self.conv = nn.Conv2d(in_chn, out_chn, ksize, 1, padding='same')
        self.bn = nn.BatchNorm2d(out_chn)
        self.max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_ksize) if max_pool_ksize else None
        self.lif = snn.Leaky(beta=beta, learn_beta=learn_beta, learn_threshold=learn_thres)

    def forward(self, x, mem):
        x = self.conv(x)
        x = self.bn(x)
        x, mem = self.lif(x, mem)
        if self.max_pool:
            x = self.max_pool(x)
        return x, mem

    def init_leaky(self):
        return self.lif.init_leaky()
