import pytest
import pytorch_lightning as pl

from snn_voice.model.hjh import HjhCNN, HjhSCNN, HjhSNN
from snn_voice.model.mx.m5 import M5CNN, M5SNN
from snn_voice.model.piczak import PiczakCNN, PiczakSNN
from snn_voice.model.tcy import TcyNN, TcySNN
from snn_voice.utils.time_step_replica import repeat_replica, rate_replica, latency_replica


@pytest.mark.parametrize('net', [
    PiczakCNN(35),
    PiczakSNN(35, 2, repeat_replica),
    PiczakSNN(35, 2, rate_replica),
    PiczakSNN(35, 2, latency_replica),
    HjhCNN(35),
    HjhSCNN(35, 2, repeat_replica),
    HjhSCNN(35, 2, rate_replica),
    HjhSCNN(35, 2, latency_replica),
    HjhSNN(35, 2, repeat_replica),
    HjhSNN(35, 2, rate_replica),
    HjhSNN(35, 2, latency_replica),
    TcyNN(35),
    TcySNN(35, 2, repeat_replica),
    TcySNN(35, 2, rate_replica),
    TcySNN(35, 2, latency_replica),
])
def test_spec_models(net, dm_spec):
    run_experiment(net, dm_spec)


@pytest.mark.parametrize('net', [
    M5CNN(35),
    M5SNN(35, 2, repeat_replica),
    M5SNN(35, 2, rate_replica),
    M5SNN(35, 2, latency_replica),
])
def test_models(net, dm):
    run_experiment(net, dm)


@pytest.mark.parametrize('net, learn_beta, learn_thres', [
    (M5SNN(35, 2, repeat_replica, learn_beta=True, learn_thres=True), True, True),
    (M5SNN(35, 2, repeat_replica, learn_beta=False, learn_thres=True), False, True),
    (M5SNN(35, 2, repeat_replica, learn_beta=True, learn_thres=False), True, False),
    (M5SNN(35, 2, repeat_replica, learn_beta=False, learn_thres=False), False, False)
])
def test_learnable_args(net, dm, learn_beta, learn_thres):
    assert net.snn[0].lif.beta.requires_grad == learn_beta
    assert net.snn[0].lif.threshold.requires_grad == learn_thres
    run_experiment(net, dm)


def run_experiment(net, dm):
    """ Runs a singular experiment for each net and DataModule """
    net = net

    trainer = pl.Trainer(
        accelerator='cpu',
        limit_train_batches=2,
        limit_val_batches=1,
        max_epochs=1,
    )
    trainer.fit(net, datamodule=dm)
    # pred = trainer.predict(net, datamodule=dm)
