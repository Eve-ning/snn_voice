from snn_voice.model.module import ModuleSNNRepeat, ModuleSNN
from snn_voice.model.mx.m5 import M5SNN


class M5SNNRepeat(ModuleSNNRepeat, ModuleSNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = M5SNN(lif_beta=self.lif_beta, n_classes=self.n_classes)
