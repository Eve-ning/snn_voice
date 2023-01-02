import snntorch as snn
import torch
from torch import nn


class PiczakSNNBlock(nn.Module):
    def __init__(self,
                 in_chn, out_chn,
                 ks_freq, ks_time,
                 stride_freq, stride_time,
                 pool_freq, pool_time,
                 pool_freq_stride=None, pool_time_stride=None,
                 lif_beta: float = 0.2):
        """

        Args:
            in_chn:
            out_chn:
            ks_freq:
            ks_time:
            stride_freq:
            stride_time:
            pool_freq:
            pool_time:
            pool_freq_stride:
            pool_time_stride:
            lif_beta:
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_chn, out_chn,
            kernel_size=(ks_freq, ks_time),
            stride=(stride_freq, stride_time)
        )
        self.bn = nn.BatchNorm2d(out_chn)
        self.lif = snn.Leaky(beta=lif_beta)
        self.max_pool = nn.MaxPool2d(
            (pool_freq, pool_time),
            (pool_freq_stride, pool_time_stride)
        )

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
        x = self.bn(x)
        x, mem = self.lif(x, mem)
        x = self.max_pool(x)
        return x, mem
