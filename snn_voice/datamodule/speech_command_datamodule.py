from pathlib import Path

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample

from snn_voice.settings import DATA_DIR, SPEECHCOMMAND_SR


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: Path = DATA_DIR,
            batch_size: int = 16,
            downsample: int = 4000
    ):
        """ Creates a DataModule to be used for fitting models through
        PyTorchLightning """
        super().__init__()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.le = LabelEncoder()
        self.downsample = Resample(SPEECHCOMMAND_SR, downsample)
        # self.prepare_data()
        # self.setup()

    def prepare_data(self):
        """ Ran once to download all data necessary before data setup """
        SPEECHCOMMANDS(self.data_dir.as_posix(), download=True)

    def setup(self, stage=""):
        """ Sets up the train, validation & test """
        self.le.fit(self.classes)
        self.train = SPEECHCOMMANDS(self.data_dir.as_posix(), subset="training")
        self.val = SPEECHCOMMANDS(self.data_dir.as_posix(), subset="validation")
        self.test = SPEECHCOMMANDS(self.data_dir.as_posix(), subset="testing")

    @property
    def classes(self):
        """ Gets the class labels from the downloaded data dir """
        return [
            d.name
            for d
            in (self.data_dir / "SpeechCommands/speech_commands_v0.02/").glob("*/")
            if "." not in d.name and
               d.name.islower() and
               not d.name.startswith("_")
        ]

    def collate_fn_factory(self):
        """ Creates the collate_fn using a factory

        Notes:
          A factory is necessary to include the self.le via a non-arg
        """

        def collate_fn(x):
            ars = [i[0].squeeze() for i in x]
            labs = [i[2] for i in x]
            lab_ixs = torch.tensor(self.le.transform(labs), dtype=int)

            ar = nn.utils.rnn.pad_sequence(ars, batch_first=True).unsqueeze(1)
            ar = self.downsample(ar)
            return (
                ar,  # ar: B x 1 x T
                lab_ixs,  # lab: 0, 1, ... , 34
                # [i[1] for i in x], # sr
                # [i[3] for i in x], # uid
                # [i[4] for i in x], # wid
            )

        return collate_fn

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn_factory())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,
                          collate_fn=self.collate_fn_factory(), shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          collate_fn=self.collate_fn_factory())

    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,
                          collate_fn=self.collate_fn_factory())
