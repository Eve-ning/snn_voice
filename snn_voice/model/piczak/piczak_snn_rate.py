from snn_voice.model.module import ModuleSNNRate, ModuleSNN
from snn_voice.model.piczak import piczak_snn_init


class PiczakSNNRate(ModuleSNNRate, ModuleSNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
