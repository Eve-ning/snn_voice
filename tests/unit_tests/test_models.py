import pytest
import pytorch_lightning as pl

from snn_voice.model.mx.m5 import M5CNN
from snn_voice.model.mx.m5 import M5SNNLatency
from snn_voice.model.mx.m5 import M5SNNRate
from snn_voice.model.mx.m5 import M5SNNRepeat
from snn_voice.model.piczak import PiczakCNN
from snn_voice.model.piczak import PiczakSNNLatency
from snn_voice.model.piczak import PiczakSNNRate
from snn_voice.model.piczak import PiczakSNNRepeat


@pytest.mark.parametrize('Model', [
    PiczakCNN,
    PiczakSNNRate,
    PiczakSNNRepeat,
    PiczakSNNLatency
])
def test_spec_models(Model, dm_spec):
    run_experiment(Model, dm_spec)


@pytest.mark.parametrize('Model', [
    M5CNN,
    M5SNNRate,
    M5SNNRepeat,
    M5SNNLatency
])
def test_models(Model, dm):
    run_experiment(Model, dm)


def run_experiment(Model, dm):
    """ Runs a singular experiment for each Model and DataModule """
    net = Model()

    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu')
    trainer.fit(net, datamodule=dm)
    pred = trainer.predict(net, datamodule=dm)
