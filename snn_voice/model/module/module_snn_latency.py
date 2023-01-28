from abc import ABC

import torch
from snntorch.spikegen import latency


class ModuleSNNLatency(ABC):

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        x = x[:, 0].abs()

        # TODO: Unsure why x is unnormalized? May be of concern
        x = (x - x.min()) / (x.max() - x.min())

        return latency(x, num_steps=n_steps, clip=True, threshold=0.01, normalize=True).unsqueeze(2)
