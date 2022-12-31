from dataclasses import dataclass, field
from typing import Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchaudio import load

from snn_voice.settings import SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES, DATA_SAMPLE_DIR
from snn_voice.utils.collate import CollateFn


class SampleDatasetBase(Dataset):
    def __init__(self, data_dir):
        self.ars_audio = []
        self.str_class = []
        for fp in data_dir.glob("**/*.wav"):
            self.ars_audio.append(load(fp)[0])
            self.str_class.append(fp.parts[-2])

    def __len__(self):
        return len(self.ars_audio)

    def __getitem__(self, ix):
        return self.ars_audio[ix], None, self.str_class[ix]


@dataclass
class SampleDataModule(pl.LightningDataModule):
    """ This is a 3-file per class sample dataset from SpeechCommands V2 """
    batch_size: int = 4
    classes: Tuple[str] = SPEECHCOMMAND_CLASSES
    data_dir = DATA_SAMPLE_DIR
    num_workers: int = 0
    dl_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        super(SampleDataModule, self).__init__()
        self.collate_fn = CollateFn(SPEECHCOMMAND_SR, SPEECHCOMMAND_CLASSES)
        ds = SampleDatasetBase(self.data_dir)
        ds_ea_size = int(len(ds) // 3)
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, (ds_ea_size,) * 3)
        self.dl_kwargs = {**self.dl_kwargs,
                          'batch_size': self.batch_size,
                          'collate_fn': self.collate_fn,
                          'num_workers': self.num_workers}

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs)
