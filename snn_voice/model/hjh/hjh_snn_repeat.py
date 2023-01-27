from snn_voice.model.module import ModuleSNNRepeat
from snn_voice.model.hjh import hjh_snn_init


class HjhSNNRepeat(ModuleSNNRepeat):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hjh_snn_init(self)
