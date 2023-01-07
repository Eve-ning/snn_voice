from snn_voice.model.module import ModuleSNNRate
from snn_voice.model.mx import m5_snn_init


class M5SNNRate(ModuleSNNRate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m5_snn_init(self)
