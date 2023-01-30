import torch

from snn_voice.model.hjh.hjh_snn import HjhSNN
from snn_voice.utils.time_step_replica import repeat_replica


class HjhSNNRepeat(HjhSNN):

    def __init__(self, n_classes: int, lif_beta: float, n_steps: int):
        super().__init__(
            n_classes=n_classes,
            lif_beta=lif_beta,
            n_steps=n_steps
        )

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        return repeat_replica(x, n_steps)
