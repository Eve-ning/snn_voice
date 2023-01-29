import torch

from snn_voice.model.hjh.hjh_snn import HjhSNN
from snn_voice.utils.time_step_replica import rate_replica


class HjhSNNRate(HjhSNN):

    def __init__(self, lif_beta: float, n_classes: int, n_steps: int):
        super().__init__(lif_beta, n_classes, n_steps)

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        return rate_replica(x, n_steps)
