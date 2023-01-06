from snn_voice.model.module.module_snn_rate import ModuleSNNRate
from snn_voice.model.piczak.piczak_snn import piczak_snn_init


class PiczakSNNRate(ModuleSNNRate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
