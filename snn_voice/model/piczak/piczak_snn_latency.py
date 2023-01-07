from snn_voice.model.module import ModuleSNNLatency
from snn_voice.model.piczak import piczak_snn_init


class PiczakSNNLatency(ModuleSNNLatency):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
