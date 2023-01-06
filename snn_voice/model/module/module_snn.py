from abc import abstractmethod, ABC

import torch

from snn_voice.model.module import Module


class ModuleSNN(Module, ABC):
    """ Defines the base SNN Module """

    def __init__(
            self,
            n_steps: int = 2,
            lif_beta: float = 0.2,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_steps = n_steps
        self.lif_beta = lif_beta

    @abstractmethod
    def time_step_replica(self, x) -> torch.Tensor:
        ...

    def forward(self, x):
        # xt: Time Step, Batch Size, Channel = 1, Sample Rate
        xt = self.time_step_replica(x)

        mems = {k: blk.init_leaky() for k, blk in self.conv_blks.named_children()}

        hist_y = []

        # For each time step
        for step in range(self.n_steps):
            x = xt[step]

            for blk_name, blk in self.conv_blks.named_children():
                # blk_name: str. E.g. 'conv_blk1'
                x, mem = blk(x, mems[blk_name])

                # Update the membrane potential.
                mems[blk_name] = mem

            # After Avg Pool
            # Spect   : BS, FT, 1, 1
            # No Spect: BS, FT, 1
            # Squeeze + Unsqueeze: BS, 1, FT
            hist_y.append(self.classifier(self.avg_pool(x).squeeze()))

        return torch.stack(hist_y, dim=0)

    def step(self, batch):
        # x: Batch Size, Channel = 1, Sample Rate
        x, y_true = batch

        # yt: Time Step, Batch Size, Classes
        yt = self(x)

        # Sum it on the time step axes & squeeze
        # y: Batch Size, Classes
        y_pred_l = yt.sum(dim=0).to(float)

        # Scale y to a proportion
        # We can consider moving the optimizer here
        # See: https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#snntorch.functional.loss.ce_count_loss
        # y_pred_l = y # / self.n_steps
        return x, y_pred_l, y_true
