from torch import nn


class MxCNNBlock(nn.Module):
    def __init__(self, in_chn, out_chn, ksize, step=1):
        super().__init__()
        self.conv = nn.Conv1d(in_chn, out_chn, ksize, step,
                              padding=int(ksize // 2 - 1))
        self.bn = nn.BatchNorm1d(out_chn)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x
