from pytorch_lightning.cli import LightningCLI

from snn_voice.datamodule import SpeechCommandsDataModule
from snn_voice.model.piczak import PiczakCNN, PiczakSNNRepeat, PiczakSNNLatency, PiczakSNNRate  # noqa: F401
from snn_voice.model.mx.m5 import M5CNN, M5SNNRepeat, M5SNNLatency, M5SNNRate  # noqa: F401


class SpeechCommandsDataModuleNoMel(SpeechCommandsDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(n_mels=None, *args, **kwargs)


cli = LightningCLI()
