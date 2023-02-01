from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import pad
from torchaudio.transforms import MelSpectrogram

from snn_voice.settings import EPSILON, MIN_WINDOW_MS


class CollateFn:
    def __init__(self, sr, classes, n_mels: int | None = 60):
        self.sr = sr
        self.classes = classes
        n_fft = int(MIN_WINDOW_MS / (1 / sr * 1000))
        self.mel_spec = MelSpectrogram(n_mels=n_mels, n_fft=n_fft) if n_mels else n_mels

    def __call__(self, batch):
        ar_list = []
        labels = []

        for b in batch:
            if b[0].shape[-1] != self.sr:
                ar = pad(b[0], [0, self.sr - b[0].shape[-1]], 'constant')
            else:
                ar = b[0]
            if self.mel_spec is not None:
                ar = self.mel_spec(ar)
                ar = np.log(ar + EPSILON)
                ar_ = ar[ar > -20]
                ar = (ar - ar_.mean()) / ar_.std()
                ar[ar < -5] = 0
            ar_list.append(ar)
            labels.append(self.classes.index(b[2]))

        return torch.stack(ar_list), torch.tensor(labels, dtype=torch.long)
