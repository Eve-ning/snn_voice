from abc import ABC

import torch

from snn_voice.model.module.module_snn import ModuleSNN


class ModuleSNNRepeat(ModuleSNN, ABC):

    def time_step_replica(self, x) -> torch.Tensor:
        return x.repeat(self.n_steps, 1, 1, 1)
