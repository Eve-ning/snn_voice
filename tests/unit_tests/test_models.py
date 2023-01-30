import pytest
import pytorch_lightning as pl

from snn_voice.model.mx.m5 import M5CNN, M5SNNLatency, M5SNNRate, M5SNNRepeat
from snn_voice.model.piczak import PiczakCNN, PiczakSNNLatency, PiczakSNNRate, PiczakSNNRepeat


@pytest.mark.parametrize('net', [
    PiczakCNN(),
    PiczakSNNRate(0.5, 35, 2),
    PiczakSNNRepeat(0.5, 35, 2),
    PiczakSNNLatency(0.5, 35, 2),
])
def test_spec_models(net, dm_spec):
    run_experiment(net, dm_spec)


@pytest.mark.parametrize('net', [
    M5CNN(),
    M5SNNRate(0.5, 35, 2),
    M5SNNRepeat(0.5, 35, 2),
    M5SNNLatency(0.5, 35, 2)
])
def test_models(net, dm):
    run_experiment(net, dm)


def run_experiment(net, dm):
    """ Runs a singular experiment for each net and DataModule """
    net = net

    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu')
    trainer.fit(net, datamodule=dm)
    pred = trainer.predict(net, datamodule=dm)
