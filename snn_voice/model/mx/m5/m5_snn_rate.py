from snn_voice.model.module import ModuleSNNRate, ModuleSNN
from snn_voice.model.mx.m5 import m5_snn_init


class M5SNNRate(ModuleSNNRate, ModuleSNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m5_snn_init(self, lif_beta=self.lif_beta, n_classes=self.n_classes)
