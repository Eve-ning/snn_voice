from torch import nn


class PiczakCNNBlock(nn.Module):
    def __init__(self, in_chn, out_chn, ksize, step, max_pool_ksize, max_pool_step, dropout=None):
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, ksize, step)
        self.max_pool = nn.MaxPool2d(max_pool_ksize, max_pool_step)
        self.dropout = nn.Dropout(dropout) if dropout else dropout
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(x)
        return x
