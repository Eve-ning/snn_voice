from dataclasses import dataclass, field
from typing import Tuple, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from src.settings import SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES, DATA_DIR
from src.utils.collate import CollateFn


@dataclass
class SpeechCommandDataset(pl.LightningDataModule):
    batch_size: int = 128
    classes: Tuple[str] = SPEECHCOMMAND_CLASSES
    data_dir = DATA_DIR / "SpeechCommands"
    download: bool = False
    num_workers: int = 0
    dl_kwargs: dict = field(default_factory=dict)
    train_ds = None
    val_ds = None
    test_ds = None

    def __post_init__(self):
        self.collate_fn = CollateFn(SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES)
        self.dl_kwargs = {**self.dl_kwargs,
                          'batch_size': self.batch_size,
                          'collate_fn': self.collate_fn,
                          'num_workers': self.num_workers}

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="training")
        self.val_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="validation")
        self.test_ds = SPEECHCOMMANDS(self.data_dir, download=self.download, subset="testing")

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs)
