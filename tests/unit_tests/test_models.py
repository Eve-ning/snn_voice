import pytest
import pytorch_lightning as pl

from snn_voice.model.hjh import HjhCNN, HjhSCNN, HjhSNN
from snn_voice.model.mx.m5 import M5CNN, M5SNN
from snn_voice.model.piczak import PiczakCNN, PiczakSNN
from snn_voice.model.tcy import TcyNN, TcySNN
from snn_voice.utils.time_step_replica import repeat_replica, rate_replica, latency_replica


@pytest.mark.parametrize('net', [
    PiczakCNN(35),
    PiczakSNN(35, 0.5, 2, repeat_replica),
    PiczakSNN(35, 0.5, 2, rate_replica),
    PiczakSNN(35, 0.5, 2, latency_replica),
    HjhCNN(35),
    HjhSCNN(35, 0.5, 2, repeat_replica),
    HjhSCNN(35, 0.5, 2, rate_replica),
    HjhSCNN(35, 0.5, 2, latency_replica),
    HjhSNN(35, 0.5, 2, repeat_replica),
    HjhSNN(35, 0.5, 2, rate_replica),
    HjhSNN(35, 0.5, 2, latency_replica),
    TcyNN(35),
    TcySNN(35, 0.5, 2, repeat_replica),
    TcySNN(35, 0.5, 2, rate_replica),
    TcySNN(35, 0.5, 2, latency_replica),
])
def test_spec_models(net, dm_spec):
    run_experiment(net, dm_spec)


@pytest.mark.parametrize('net', [
    M5CNN(35),
    M5SNN(35, 0.5, 2, repeat_replica),
    M5SNN(35, 0.5, 2, rate_replica),
    M5SNN(35, 0.5, 2, latency_replica),
])
def test_models(net, dm):
    run_experiment(net, dm)


def run_experiment(net, dm):
    """ Runs a singular experiment for each net and DataModule """
    net = net

    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu')
    trainer.fit(net, datamodule=dm)
    pred = trainer.predict(net, datamodule=dm)
