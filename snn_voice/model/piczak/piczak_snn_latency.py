from snn_voice.model.module import ModuleSNNLatency, ModuleSNN
from snn_voice.model.piczak import PiczakSNN


class PiczakSNNLatency(ModuleSNNLatency, ModuleSNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = PiczakSNN(lif_beta=self.lif_beta, n_classes=self.n_classes)
