import torch

from snn_voice.model.piczak import PiczakSNN
from snn_voice.utils.time_step_replica import repeat_replica


class PiczakSNNRepeat(PiczakSNN):

    def __init__(self, lif_beta: float, n_classes: int, n_steps: int):
        super().__init__(lif_beta, n_classes, n_steps)

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        return repeat_replica(x, n_steps)
