from snn_voice.model.module.module_snn_repeat import ModuleSNNRepeat
from snn_voice.model.piczak.piczak_snn import piczak_snn_init


class PiczakSNNRepeat(ModuleSNNRepeat):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
