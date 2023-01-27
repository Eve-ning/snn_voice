from snn_voice.model.module import ModuleSNNLatency
from snn_voice.model.hjh import hjh_snn_init


class HjhSNNLatency(ModuleSNNLatency):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hjh_snn_init(self)
