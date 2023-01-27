from torch import nn


class HjhCNNBlock(nn.Module):
    def __init__(self, in_chn, out_chn, ksize, max_pool_ksize=None):
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
