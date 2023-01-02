import torch.nn as nn


class PiczakCNNBlock(nn.Module):
    def __init__(
            self,
            in_chn, out_chn,
            ks_freq, ks_time,
            stride_freq, stride_time,
            pool_freq, pool_time,
            pool_freq_stride=None, pool_time_stride=None
    ):
        """ This implements the M5 block in 2D """
        super().__init__()

        self.conv = nn.Conv2d(
            in_chn, out_chn,
            kernel_size=(ks_freq, ks_time),
            stride=(stride_freq, stride_time)
        )
        self.bn = nn.BatchNorm2d(out_chn)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(
            (pool_freq, pool_time),
            (pool_freq_stride, pool_time_stride)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x
