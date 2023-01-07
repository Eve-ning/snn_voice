from snn_voice.model.module.module_snn_latency import ModuleSNNLatency
from snn_voice.model.mx.m5_snn import m5_snn_init


class M5SNNLatency(ModuleSNNLatency):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m5_snn_init(self)
