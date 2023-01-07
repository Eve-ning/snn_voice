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
        """ This expands x's shape in the 1st dim for the forward function loop """
        ...

    def forward(self, x):
        # xt: Time Step, Batch Size, Channel = 1, Sample Rate
        xt = self.time_step_replica(x)

        mems = {k: blk.init_leaky() for k, blk in self.conv_blks.named_children()}

        yt_list = []

        # For each time step
        for step in range(self.n_steps):
            # Given that the 1st dim of xt is time step
            # x: Batch Size, Channel = 1, Sample Rate
            x = xt[step]

            # For each conv_blk, we have a membrane potential:
            #   It is stored in the `mems` variable as {blk_name: blk_mem}
            # Each time step should update each mem block through the blk_name
            # * Note that we shouldn't depend on the `mems` during the loop as we're mutating it.
            for blk_name, blk in self.conv_blks.named_children():
                # blk_name: str. E.g. 'conv_blk1'
                # blk: nn.Module. E.g. M5SNNBlock
                x, mem = blk(x, mems[blk_name])

                # Update the membrane potential.
                mems[blk_name] = mem

            # After Avg Pool
            # Spect   : BS, FT, 1, 1
            # No Spect: BS, FT, 1
            # Squeeze + Unsqueeze: BS, 1, FT
            yt_list.append(self.classifier(self.avg_pool(x).squeeze()))

        # This will yield the time-step y
        # yt: Time Step, Batch Size, Channel = 1, Feature
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
