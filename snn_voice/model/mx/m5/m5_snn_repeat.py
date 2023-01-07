from snn_voice.model.module import ModuleSNNRepeat
from snn_voice.model.mx.m5 import m5_snn_init


class M5SNNRepeat(ModuleSNNRepeat):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m5_snn_init(self)
