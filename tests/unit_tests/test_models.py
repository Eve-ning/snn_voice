import pytest
import pytorch_lightning as pl
import torch

from snn_voice.datamodule.sample_datamodule import SampleDataModule
from snn_voice.datamodule.speech_command_datamodule import SpeechCommandsDataModule
from snn_voice.model.cnn_m5 import CnnM5
from snn_voice.model.cnn_piczak import CnnPiczak
from snn_voice.model.lit_wrapper import LitWrapper
from snn_voice.model.mx.m5_cnn import M5CNN
from snn_voice.model.mx.m5_snn_latency import M5SNNLatency
from snn_voice.model.mx.m5_snn_rate import M5SNNRate
from snn_voice.model.mx.m5_snn_repeat import M5SNNRepeat
from snn_voice.model.snn_tcy import SnnTCY
from snn_voice.model.srnn_hjh import SrnnHJH


@pytest.mark.parametrize(
    'Model',
    [M5CNN, M5SNNRate, M5SNNRepeat, M5SNNLatency]
)
def test_models(Model, dm):
    net = Model()

    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu')
    trainer.fit(net, datamodule=dm)
    pred = trainer.predict(net, datamodule=dm)
