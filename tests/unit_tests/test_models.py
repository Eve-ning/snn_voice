import pytest
import pytorch_lightning as pl

from snn_voice.model.mx.m5_cnn import M5CNN
from snn_voice.model.mx.m5_snn_latency import M5SNNLatency
from snn_voice.model.mx.m5_snn_rate import M5SNNRate
from snn_voice.model.mx.m5_snn_repeat import M5SNNRepeat
from snn_voice.model.piczak.piczak_cnn import PiczakCNN
from snn_voice.model.piczak.piczak_snn_latency import PiczakSNNLatency
from snn_voice.model.piczak.piczak_snn_rate import PiczakSNNRate
from snn_voice.model.piczak.piczak_snn_repeat import PiczakSNNRepeat


@pytest.mark.parametrize(
    'Model',
    [M5CNN, M5SNNRate, M5SNNRepeat, M5SNNLatency,
     PiczakCNN, PiczakSNNRate, PiczakSNNRepeat, PiczakSNNLatency]
)
def test_models(Model, dm):
    net = Model()

    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu')
    trainer.fit(net, datamodule=dm)
    pred = trainer.predict(net, datamodule=dm)
