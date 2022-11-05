from typing import Tuple

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, Resample


def m5_2d_block(
        in_dim, out_dim,
        ks_freq, ks_time,
        stride_freq, stride_time,
        max_pool_freq, max_pool_time,
        max_pool_freq_stride=None, max_pool_time_stride=None
):
    """ This implements the M5 block in 2D """
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim,
                  kernel_size=(ks_freq, ks_time),
                  stride=(stride_freq, stride_time)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((max_pool_freq, max_pool_time),
                     (max_pool_freq_stride, max_pool_time_stride))
    )


class CnnMel(nn.Module):
    def __init__(
            self,
            n_classes: int,
            resample: Tuple[int, int] = (16000, 8000),
            n_freq: int = 60,
            n_time: int = 41,
            n_channel=80,
            *args,
            **kwargs
    ):
        """ Implements the MelSpectrogram extraction before the CNN blocks

        Notes:
            This references the paper:
            Piczak, Karol J. "Environmental sound classification with convolutional neural networks."
            2015 IEEE 25th international workshop on machine learning for signal processing (MLSP). IEEE, 2015.

        Args:
            n_classes: Number of classes to output
            resample: The initial and the resulting sample rate as a tuple
            n_freq: Number of mel scale frequency bins
            n_time: Number of time bins
            n_channel: Number of channels to use for intermediate CNN blocks.
        """
        super(CnnMel, self).__init__()
        n_fft = int(resample[1] / (n_time - 1) * 2)

        self.feature_extraction = nn.Sequential(
            Resample(resample[0], resample[1]),
            MelSpectrogram(sample_rate=resample[1],
                           n_fft=n_fft,
                           normalized=True,
                           norm="slaney",
                           n_mels=n_freq),
            m5_2d_block(1, n_channel, 57, 6, 1, 1, 4, 3, 1, 3),
            m5_2d_block(n_channel, n_channel, 1, 3, 1, 1, 1, 3, 1, 3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = torch.log(self.feature_extraction(x) + 1e-3)
        x = (x - x.mean()) / (x.std())
        x = self.flatten(x)
        x = self.classifier(x)
        return x
