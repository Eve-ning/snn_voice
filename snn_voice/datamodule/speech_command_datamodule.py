from pathlib import Path

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import load
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
        self.prepare_data()
        self.setup()

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

    def load_samples(self, subset: str = "validation"):
        """ Loads a single sample for each class from the dataset

        Args:
            subset: The subset string, can be "validation", "testing"

        Returns:
            A dictionary of key: utterance label and torch.Tensor
        """

        def load_and_pad(path: Path):
            ar = load(path)[0].unsqueeze(0)

            if ar.shape[-1] != SPEECHCOMMAND_SR:
                # Pad those samples that are smaller than SAMPLE_RATE for some reason
                short = SPEECHCOMMAND_SR - ar.shape[-1]
                ar = nn.functional.pad(ar, (short // 2, short - short // 2))

            return self.downsample(ar)

        speech_commands_path = Path(f"{DATA_DIR.as_posix()}/SpeechCommands/speech_commands_v0.02/")
        list_path = speech_commands_path / f"{subset}_list.txt"
        samples = {}

        with open(list_path, "r") as f:
            sample_paths = {}
            while line := f.readline():
                if line:
                    sample_paths[line.split("/")[0]] = line.split("/")[1].strip()

            for sample_name, sample_fn in sample_paths.items():
                samples[sample_name] = load_and_pad(speech_commands_path / sample_name / sample_fn)

        return samples
