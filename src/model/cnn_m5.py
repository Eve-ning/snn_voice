import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from src.settings import SPEECHCOMMAND_SR


def m5_block(in_dim, out_dim, ks, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=ks, stride=stride),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(4)
    )


class CnnM5(nn.Module):
    def __init__(
            self,
            n_classes: int,
            downsample: int = 8000,
            n_input=1,
            stride=16,
            n_channel=32
    ):
        """ Implements the M5 Model from the paper in the docstring.

        Notes:
            This adjusts the implementation slightly for the purpose of this project

            https://arxiv.org/pdf/1610.00087.pdf
            Dai, Wei, et al. "Very deep convolutional neural networks for raw waveforms."
            2017 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2017.

        Args:
            n_classes: Number of classes to output
            downsample: Sample Rate to use
            n_input: Number of input channels (Mono: 1, Dual: 2)
            stride: Stride length for the first M5 Block
            n_channel: Number of channels to use for intermediate CNN blocks.

        """
        super(CnnM5, self).__init__()

        self.feature_extraction = nn.Sequential(
            Resample(SPEECHCOMMAND_SR, downsample),
            m5_block(n_input, n_channel, ks=80, stride=stride),
            m5_block(n_channel, n_channel, ks=3),
            m5_block(n_channel, n_channel * 2, ks=3),
            m5_block(n_channel * 2, n_channel * 2, ks=3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * n_channel, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)

        # We expect a shape of (BS, 64, X) X is variable
        # To solve the variable issue, we average the last dimension
        x = F.avg_pool1d(x, x.shape[-1])

        x = self.classifier(x)
        return x
