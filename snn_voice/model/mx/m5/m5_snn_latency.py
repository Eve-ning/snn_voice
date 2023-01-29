import torch

from snn_voice.model.mx.m5 import M5SNN
from snn_voice.utils.time_step_replica import latency_replica


class M5SNNLatency(M5SNN):

    def __init__(self, lif_beta: float, n_classes: int, n_steps: int):
        super().__init__(lif_beta, n_classes, n_steps)

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        return latency_replica(x, n_steps)
