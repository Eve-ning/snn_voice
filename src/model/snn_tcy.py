from typing import Tuple

import snntorch as snn
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, Resample

from src.settings import EPSILON


class SnnTCY(nn.Module):
    def __init__(
            self,
            n_classes: int,
            resample: Tuple[int, int] = (16000, 8000),
            n_freq: int = 60,
            n_time: int = 41,
            leaky_beta=0.85,
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
        super(SnnTCY, self).__init__()
        n_fft = int(resample[1] / (n_time - 1) * 2)

        self.feature_extraction = nn.Sequential(
            Resample(resample[0], resample[1]),
            MelSpectrogram(sample_rate=resample[1],
                           n_fft=n_fft,
                           normalized=True,
                           norm="slaney",
                           n_mels=n_freq),
        )
        self.spike_gen = snn.Leaky(beta=leaky_beta, init_hidden=True)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(n_freq * n_time, n_classes, ),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = torch.log(self.feature_extraction(x) + EPSILON)
        x = (x - x.mean()) / (x.std())
        x = self.spike_gen(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# from src.utils.load import load_any_sample
# import matplotlib.pyplot as plt
# import seaborn as sns
# ar_audio, sr = load_any_sample(ix=1, sample_name="backward")
# ar_audio = ar_audio.reshape(1, 1, -1)
# ar_out = SnnTCY(35, leaky_beta=1)(ar_audio)

# sns.heatmap(ar_out.squeeze().detach().numpy())
# plt.ylabel("Frequency Bins")
# plt.xlabel("Time Bins")
# plt.tight_layout()
# plt.show()
