from abc import ABC

import torch

from snn_voice.model.module import ModuleSNN


class ModuleSNNRepeat(ModuleSNN, ABC):

    def time_step_replica(self, x) -> torch.Tensor:
        # This dynamically repeats on a new axis (at the front).
        # E.g., if x.shape = (A, B, C)
        #   x.repeat(n_steps, 1, 1, 1) <- the number of 1s are dynamically yield from the shape
        #   The number of 1s is dynamic as x has varying ndims due to non-spec and spectrogram transforms
        return x.repeat(self.n_steps, *(1,) * x.ndim)
