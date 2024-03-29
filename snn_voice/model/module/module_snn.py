from abc import abstractmethod, ABC
from typing import Callable, List

import torch

from snn_voice.model.module import Module


class ModuleSNN(Module, ABC):
    """ Defines the base SNN Module """

    def __init__(
            self,
            n_steps: int,
            time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_steps = n_steps
        self.time_step_replica_ = time_step_replica

    def time_step_replica(self, x, n_steps: int) -> torch.Tensor:
        """ This expands x's shape in the 1st dim for the forward function loop """
        return self.time_step_replica_(x, n_steps)

    @abstractmethod
    def time_step_forward(self, x, mems: List[torch.Tensor]) -> torch.Tensor:
        ...

    @abstractmethod
    def init_mems(self) -> List[torch.Tensor]:
        """ Initialized Membrane for SNN Layers """
        ...

    def forward(self, x):
        # xt: Time Step, Batch Size, Channel = 1, Sample Rate
        xt = self.time_step_replica(x, self.n_steps)

        # yt: Time Step, Batch Size, Channel = 1, Feature
        yt_list = []
        mems = self.init_mems()

        for step in range(self.n_steps):
            x = xt[step]
            y = self.time_step_forward(x, mems)
            yt_list.append(y)

        for mem in mems:
            mem.detach()

        yt = torch.stack(yt_list, dim=0)
        return yt

    def step(self, batch):
        # x: Batch Size, Channel = 1, Sample Rate
        x, y_true = batch

        # yt: Time Step, Batch Size, Classes
        yt = self(x)

        # Sum it on the time step axes & squeeze
        # y: Batch Size, Classes
        y_pred_l = yt.sum(dim=0).to(float)

        # Scale y to a proportion
        # We can consider moving the optimizer here, but I'm unsure if it's stable
        # See: https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#snntorch.functional.loss.ce_count_loss
        # y_pred_l = y # / self.n_steps
        return x, y_pred_l, y_true
