from abc import ABC

import torch
from snntorch.spikegen import latency

from snn_voice.model.mx.mx_snn import MxSNN


class MxSNNLatency(MxSNN, ABC):

    def time_step_replica(self, x) -> torch.Tensor:
        x = x[:, 0].abs()
        x = (x - x.min()) / (x.max() - x.min())
        return latency(x, num_steps=self.n_steps, clip=True,
                       threshold=0.01, normalize=True).unsqueeze(2)
