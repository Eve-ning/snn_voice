from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from src.settings import SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES, DATA_DIR


def collate_fn(batch):
    ars_audio = []
    ars_label = []

    for b in batch:
        if b[0].shape[-1] != SPEECHCOMMAND_SR:
            ar_audio = pad(b[0], [0, SPEECHCOMMAND_SR - b[0].shape[-1]], 'constant')
        else:
            ar_audio = b[0]
        ars_audio.append(ar_audio)
        ars_label.append(SPEECHCOMMAND_CLASSES.index(b[2]))

    return torch.stack(ars_audio), torch.tensor(ars_label, dtype=torch.long)


@dataclass
class SpeechCommandDataset:
    batch_size: int = 128
    classes: Tuple[str] = SPEECHCOMMAND_CLASSES
    data_dir = DATA_DIR / "SpeechCommands"
    download: bool = False
    num_workers: int = 0
    dl_kwargs: dict = field(default_factory=dict)

    def train_dl(self):
        train_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="training")
        return DataLoader(train_ds,
                          batch_size=self.batch_size,
                          collate_fn=collate_fn,
                          num_workers=self.num_workers,
                          shuffle=True,
                          **self.dl_kwargs)

    def val_dl(self):
        val_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="validation")
        return DataLoader(val_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn,
                          **self.dl_kwargs)

    def test_dl(self):
        test_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="testing")
        return DataLoader(test_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn,
                          **self.dl_kwargs)

    def dls(self):
        return self.train_dl(), self.val_dl(), self.test_dl()
