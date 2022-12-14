from snn_voice.model.module import ModuleSNNRate
from snn_voice.model.piczak import piczak_snn_init


class PiczakSNNRate(ModuleSNNRate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
