import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class CNNMel(nn.Module):
    def __init__(
            self, n_classes: int,
            seq_size: int = 16000,
            mel_bins: int = 32,
            n_fft: int = 1024
    ):
        """ Implements the MelSpectrogram extraction before the CNN blocks """
        super(CNNMel, self).__init__()

        mel_spectrogram = MelSpectrogram(
            sample_rate=seq_size,
            n_fft=n_fft,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True,
            norm="slaney",
            n_mels=mel_bins,
            mel_scale="htk",
        )

        self.feature_extraction = nn.Sequential(
            mel_spectrogram,
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
