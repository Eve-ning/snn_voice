import pytest

from snn_voice.datamodule import SampleDataModule


@pytest.fixture(scope="session")
def dm():
    """ Initializes a DataModule without a spectrogram transform """
    return SampleDataModule(n_mels=None)


@pytest.fixture(scope="session")
def dm_spec():
    """ Initializes a DataModule with a spectrogram transform """
    return SampleDataModule()
