from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram

from src.settings import SEQ_SIZE, WITH_MEL_SPEC, SPEECHCOMMAND_CLASSES, MEL_BINS, N_FFT

mel_spectrogram = MelSpectrogram(
    sample_rate=SEQ_SIZE,
    n_fft=N_FFT,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True,
    norm="slaney",
    n_mels=MEL_BINS,
    mel_scale="htk",
)


def collate_fn(batch):
    ars_audio = []
    ars_label = []

    for b in batch:
        ar_audio = pad(b[0], [0, SEQ_SIZE - b[0].shape[-1]], 'constant')
        if WITH_MEL_SPEC:
            ar_audio = mel_spectrogram(ar_audio)
        ars_audio.append(ar_audio)
        ars_label.append(SPEECHCOMMAND_CLASSES.index(b[2]))

    return torch.stack(ars_audio), torch.tensor(ars_label, dtype=torch.long)


@dataclass
class SpeechCommandDataset:
    batch_size: int = 2
    classes: Tuple[str] = SPEECHCOMMAND_CLASSES

    def train_dl(self, download: bool = False, num_workers: int = 1):
        train_ds = SPEECHCOMMANDS("data/", download=download, subset="training")
        return DataLoader(train_ds,
                          batch_size=self.batch_size,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          shuffle=True)

    def val_dl(self, download: bool = False, num_workers: int = 1):
        val_ds = SPEECHCOMMANDS("data/", download=download, subset="validation")
        return DataLoader(val_ds,
                          batch_size=self.batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn)

    def test_dl(self, download: bool = False, num_workers: int = 1):
        test_ds = SPEECHCOMMANDS("data/", download=download, subset="testing")
        return DataLoader(test_ds,
                          batch_size=self.batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn)

    def dls(self, download: bool = False, num_workers: int = 1):
        return self.train_dl(download, num_workers), \
               self.val_dl(download, num_workers), \
               self.test_dl(download, num_workers)
