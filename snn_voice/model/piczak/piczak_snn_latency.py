from snn_voice.model.module import ModuleSNNLatency, ModuleSNN
from snn_voice.model.piczak import piczak_snn_init


class PiczakSNNLatency(ModuleSNNLatency, ModuleSNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self, lif_beta=self.lif_beta, n_classes=self.n_classes)
