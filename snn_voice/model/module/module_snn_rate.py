from abc import ABC

import torch
from snntorch.spikegen import rate


class ModuleSNNRate(ABC):

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        x = x[:, 0].abs()
        x = (x - x.min()) / (x.max() - x.min())
        return rate(x, num_steps=n_steps).unsqueeze(2)
