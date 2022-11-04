from dataclasses import dataclass, field
from typing import Tuple

from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from src.settings import SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES, DATA_DIR
from src.utils.collate import CollateFn


@dataclass
class SpeechCommandDataset:
    batch_size: int = 128
    classes: Tuple[str] = SPEECHCOMMAND_CLASSES
    data_dir = DATA_DIR / "SpeechCommands"
    download: bool = False
    num_workers: int = 0
    dl_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.collate_fn = CollateFn(SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES)
        self.dl_kwargs = {**self.dl_kwargs,
                          'batch_size': self.batch_size,
                          'collate_fn': self.collate_fn,
                          'num_workers': self.num_workers}

    def train_dl(self):
        train_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="training")
        return DataLoader(train_ds, shuffle=True, **self.dl_kwargs)

    def val_dl(self):
        val_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="validation")
        return DataLoader(val_ds, **self.dl_kwargs)

    def test_dl(self):
        test_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="testing")
        return DataLoader(test_ds, **self.dl_kwargs)

    def dls(self):
        return self.train_dl(), self.val_dl(), self.test_dl()
