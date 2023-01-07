from abc import ABC

import torch
from snntorch.spikegen import rate

from snn_voice.model.module.module_snn import ModuleSNN


class ModuleSNNRate(ModuleSNN, ABC):

    def time_step_replica(self, x) -> torch.Tensor:
        x = x[:, 0].abs()
        x = (x - x.min()) / (x.max() - x.min())
        return rate(x, num_steps=self.n_steps).unsqueeze(2)
