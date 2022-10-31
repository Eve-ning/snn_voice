import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, Resample

from src.settings import SPEECHCOMMAND_SR


def cnn_block(
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
        nn.ReLU(),
        nn.MaxPool2d((max_pool_freq, max_pool_time),
                     (max_pool_freq_stride, max_pool_time_stride))
    )


class CNNMel2(nn.Module):
    def __init__(
            self,
            n_classes: int,
            downsample: int = 8000,
            n_freq: int = 60,
            n_time: int = 41,
            fs: int = 10,
            n_input=1,
            n_channel=32
    ):
        """ Implements the MelSpectrogram extraction before the CNN blocks """
        super(CNNMel2, self).__init__()
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

        self.preprocess = nn.Sequential(
            Resample(SPEECHCOMMAND_SR, downsample),
            mel_spectrogram,
        )
        self.freq_extraction = nn.Sequential(
            cnn_block(n_input, n_channel, 5, 1, 1, 1, 2, 1, 2, 1),
            cnn_block(n_channel, n_channel, 5, 1, 1, 1, 2, 1, 2, 1),
            # cnn_block(n_channel, n_channel, 5, 1, 1, 1, 2, 1, 2, 1),
            nn.AdaptiveAvgPool2d((1, fs))
        )
        self.time_extraction = nn.Sequential(
            cnn_block(n_input, n_channel, 1, 20, 1, 1, 1, 2, 1, 2),
            cnn_block(n_channel, n_channel, 1, 20, 1, 1, 1, 2, 1, 2),
            # cnn_block(n_channel, n_channel, 1, 5, 1, 1, 1, 2, 1, 2),
            nn.AdaptiveAvgPool2d((fs, 1))
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(n_channel * 2 * fs, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = torch.log(self.preprocess(x) + 1e-3)
        x = (x - x.mean()) / (x.std())
        x_freq = self.freq_extraction(x)
        x_time = self.time_extraction(x).permute(0, 1, 3, 2)
        x = torch.concat([x_freq, x_time], dim=1)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
# import torch
# CNNMel2(35, n_time=100, n_freq=32)(torch.ones([3, 1, 16000])).shape
