import pytest

from snn_voice.datamodule.speech_command_datamodule import SpeechCommandsDataModule


@pytest.fixture(scope="session")
def dm():
    """ Initializes a DataModule without a spectrogram transform """
    return SpeechCommandsDataModule(n_mels=None, batch_size=2)


@pytest.fixture(scope="session")
def dm_spec():
    """ Initializes a DataModule with a spectrogram transform """
    return SpeechCommandsDataModule(n_mels=60, batch_size=2)
