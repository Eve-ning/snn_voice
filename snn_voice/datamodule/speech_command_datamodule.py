from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import load
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample, MelSpectrogram

from snn_voice.settings import DATA_DIR, SPEECHCOMMAND_SR, MIN_WINDOW_MS

EPSILON = np.finfo(np.float64).eps


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: Path = DATA_DIR,
            batch_size: int = 16,
            downsample: Tuple = (SPEECHCOMMAND_SR, 4000),
            n_mels: int | None = 60,
    ):
        """ Creates a DataModule to be used for fitting models through
        PyTorchLightning

        Notes:
            With Mel Spectrogram:
                [Batch Size, 1, Mel Bands, Time Bands]
            Without Mel Spectrogram:
                [Batch Size, 1, Time]

            The size of the Spectrogram FFT is automatically inferred from the minimum window size (MIN_WINDOW_MS).

        Args:
            data_dir: Data Directory to download speech command data.
            batch_size: Batch size for all DataLoaders
            downsample: Downsample Resampling Rate.
            n_mels: Number of Mel Bands. If None, then Spectrogram transformation will be skipped.

        """

        super().__init__()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.le = LabelEncoder()
        self.downsample = Resample(*downsample)

        n_fft = int(MIN_WINDOW_MS / (1 / downsample[1] * 1000))

        # We'll figure out the FFT window needed to satisfy the minimum window ms
        self.mel_spec = MelSpectrogram(n_mels=n_mels, n_fft=n_fft) if n_mels else n_mels
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
            # E.g. i.shape = (1, 16000)
            ars = [i[0].squeeze() for i in x]

            # E.g. labs = ['two', 'yes']
            labs = [i[2] for i in x]
            lab_ixs = torch.tensor(self.le.transform(labs), dtype=int)

            # Some elements in ars may not be 16000, thus we need to pad it
            # E.g. [(16000, ), (15800, ), (..., )]
            #   -> Pad -> (Batch Size, 16000)
            #   -> Unsqueeze for Channel -> (Batch Size, 1, 16000)
            ar = nn.utils.rnn.pad_sequence(ars, batch_first=True).unsqueeze(1)

            # Downsample: (Batch Size, 1, 16000) -> (Batch Size, 1, 4000)
            ar = self.downsample(ar)
            ar = (ar - ar.mean()) / ar.std()

            if self.mel_spec is not None:
                ar = self.mel_spec(ar)
                ar = np.log(ar + EPSILON)
                ar_ = ar[ar > -20]
                ar = (ar - ar_.mean()) / ar_.std()
                ar[ar < -5] = 0

            # ar: (Batch Size, 1, Mel Bands, Time Bands)
            #   : (Batch Size, 1, Time)
            return (
                ar,  # ar: B x 1 x T or B x 1 x M x T
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

    @property
    def speech_commands_path(self):
        return Path(self.data_dir / f"SpeechCommands/speech_commands_v0.02/")

    def load_samples(self, subset: str = "validation"):
        """ Loads a single sample for each class from the dataset

        Args:
            subset: The subset string, can be "validation", "testing"

        Returns:
            A dictionary of key: utterance label and torch.Tensor
        """

        subset_list_path = self.speech_commands_path / f"{subset}_list.txt"
        samples = {}

        with open(subset_list_path, "r") as f:
            # Collect a unique path to sample for each class
            sample_paths = {}

            while line := f.readline():
                if line:
                    sample_paths[line.split("/")[0]] = line.split("/")[1].strip()

            ars = self.collate_fn_factory()(
                # Loads each
                [(*load(self.speech_commands_path / sample_name / sample_fn), sample_name)
                 for sample_name, sample_fn in sample_paths.items()]
            )[0]
            for ar, sample_name in zip(ars, sample_paths.keys()):
                samples[sample_name] = ar.unsqueeze(0)

        return samples
