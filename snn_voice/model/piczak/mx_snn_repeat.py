from abc import ABC

import torch

from snn_voice.model.mx.mx_snn import MxSNN


class MxSNNRepeat(MxSNN, ABC):

    def time_step_replica(self, x) -> torch.Tensor:
        return x.repeat(self.n_steps, 1, 1, 1)
