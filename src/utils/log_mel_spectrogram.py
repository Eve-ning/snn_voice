import torch
from torchaudio.transforms import MelSpectrogram

from src.settings import EPSILON


class LogMelSpectrogram(MelSpectrogram):
    def __init__(self, *args, **kwargs):
        kwargs['normalized'] = True
        kwargs['norm'] = "slaney"
        super(LogMelSpectrogram, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        x = super(LogMelSpectrogram, self).__call__(*args, **kwargs)
        return torch.log(x + EPSILON)
