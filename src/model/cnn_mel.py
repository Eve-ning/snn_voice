import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, Resample

from src.settings import SPEECHCOMMAND_SR


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


class CNNMel(nn.Module):
    def __init__(
            self,
            n_classes: int,
            downsample: int = 8000,
            n_freq: int = 60,
            n_time: int = 41,
            n_input=1,
            n_channel=80
    ):
        """ Implements the MelSpectrogram extraction before the CNN blocks

        Notes:
            This references the paper:
            Piczak, Karol J. "Environmental sound classification with convolutional neural networks."
            2015 IEEE 25th international workshop on machine learning for signal processing (MLSP). IEEE, 2015.

         """
        super(CNNMel, self).__init__()
        n_fft = int(downsample / (n_time - 1) * 2)
        mel_spectrogram = MelSpectrogram(
            sample_rate=downsample,
            n_fft=n_fft,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True,
            norm="slaney",
            n_mels=n_freq,
            mel_scale="htk",
        )

        self.feature_extraction = nn.Sequential(
            Resample(SPEECHCOMMAND_SR, downsample),
            mel_spectrogram,
            m5_2d_block(n_input, n_channel, 57, 6, 1, 1, 4, 3, 1, 3),
            m5_2d_block(n_channel, n_channel, 1, 3, 1, 1, 1, 3, 1, 3),
            # m5_2d_block(n_channel, n_channel, 1, 3, 1, 1, 2, 2),
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
#
#
# import torch
#
# CNNMel(35, n_time=81)(torch.ones([3, 1, 16000])).shape
