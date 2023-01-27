from snn_voice.model.module import ModuleSNNRate
from snn_voice.model.hjh import hjh_snn_init


class HjhSNNRate(ModuleSNNRate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hjh_snn_init(self)
