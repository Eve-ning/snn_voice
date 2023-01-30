from torch import nn


class HjhCNNBlock(nn.Module):
    def __init__(self,
                 in_chn: int,
                 out_chn: int,
                 ksize: int,
                 max_pool_ksize: int = None):
        """ A single CNN Block to be used in Hjh Models

        Args:
            in_chn: Ingoing Channels for Conv2d
            out_chn: Outgoing Channels for Conv2d
            ksize: Kernel Size for Conv2d
            max_pool_ksize: Max Pool Kernel Size, step size will be the same
        """
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, ksize, 1, padding='same')
        self.bn = nn.BatchNorm2d(out_chn)
        self.max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_ksize) if max_pool_ksize else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.max_pool:
            x = self.max_pool(x)
        return x
