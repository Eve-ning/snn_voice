import snntorch as snn
import torch
from torch import nn


class HjhSNNBlock(nn.Module):
    def __init__(self, in_chn, out_chn, ksize, step, max_pool_ksize, max_pool_step,
                 lif_beta: float, dropout=0.5):
        super().__init__()

        self.conv = nn.Conv2d(in_chn, out_chn, ksize, step)
        self.max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_step)
        self.dropout = nn.Dropout(dropout)
        self.lif = snn.Leaky(beta=lif_beta)

    def init_leaky(self) -> torch.Tensor:
        """ Initializes the Leaky & Fire block """
        return self.lif.init_leaky()

    def forward(self, x, mem):
        """ Block Forward

        Args:
            x: LIF Spikes (After Pool)
            mem: LIF Membrane Potential. Use `init_leaky` to yield initial mem

        Returns:
            The LIF Spikes and LIF Membrane as a tuple.
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x, mem = self.lif(x, mem)
        return x, mem
