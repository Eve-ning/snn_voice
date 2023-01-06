from snn_voice.model.module.module_snn_latency import ModuleSNNLatency
from snn_voice.model.piczak.piczak_snn import piczak_snn_init


class PiczakSNNLatency(ModuleSNNLatency):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
