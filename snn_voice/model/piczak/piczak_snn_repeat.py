from snn_voice.model.module import ModuleSNNRepeat, ModuleSNN
from snn_voice.model.piczak import piczak_snn_init


class PiczakSNNRepeat(ModuleSNNRepeat, ModuleSNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        piczak_snn_init(self)
