# TODO: Migrate HJ's Model
from typing import Tuple

import snntorch as snn
import torch
from torch import nn
from torchaudio.transforms import Resample

from snn_voice.settings import EPSILON
from snn_voice.utils.log_mel_spectrogram import LogMelSpectrogram


class SrnnHJH(nn.Module):
    def __init__(
            self,
            n_classes: int,
            resample: Tuple[int, int] = (16000, 4000),
            n_freq: int = 16,
            n_time: int = 48,
            leaky_beta=0.85,
            lstm_n_layers: int = 1,
            lstm_n_hidden: int = 12,
            *args,
            **kwargs
    ):
        """ Implements the SNN model proposed by Tan Chiah Ying

        Args:
            n_classes: Number of classes to output
            resample: The initial and the resulting sample rate as a tuple
            n_freq: Number of mel scale frequency bins
            n_time: Number of time bins
        """
        super(SrnnHJH, self).__init__()
        n_fft = int(resample[1] / (n_time - 1) * 2)

        self.n_time = n_time
        self.lstm_n_hidden = lstm_n_hidden
        self.lstm_n_layers = lstm_n_layers

        self.preprocessing = nn.Sequential(
            Resample(resample[0], resample[1]),
            LogMelSpectrogram(sample_rate=resample[1],
                              n_fft=n_fft,
                              n_mels=n_freq),
        )
        self.spike_gen = snn.Leaky(beta=leaky_beta, init_hidden=True)
        self.bilstm = nn.LSTM(
            input_size=n_freq,
            hidden_size=lstm_n_hidden,
            num_layers=lstm_n_layers,
            bidirectional=True,
            batch_first=True
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(lstm_n_hidden * 2, n_classes, ),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = torch.log(self.preprocessing(x) + EPSILON)
        x = (x - x.mean()) / (x.std())
        x = self.spike_gen(x)
        x, _ = self.bilstm(x.sum(dim=1).permute(0, 2, 1))
        x = self.flatten(x[:, -1])
        x = self.classifier(x)
        return x
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# ar_audio, sr = load_any_sample(ix=1, sample_name="backward")
# ar_audio = ar_audio.reshape(1, 1, -1)
# bs = 1
# ar_audio = torch.rand(bs, 1, 16000)
# model = SrnnHJH(35, leaky_beta=1, lstm_n_layers=1)
# ar_out = model(ar_audio)
# print(ar_out.shape)
# %%
#
# sns.heatmap(ar_out.detach().numpy()[0])
# plt.ylabel("Frequency Bins")
# plt.xlabel("Time Bins")
# plt.tight_layout()
# plt.show()
