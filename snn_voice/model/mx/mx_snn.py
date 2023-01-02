from abc import ABC, abstractmethod

import torch

from snn_voice.model.mx.mx_common import MxCommon


class MxSNN(MxCommon, ABC):
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

    def forward(self, x):
        """ We should expect a T x B x t input"""

        # xt: Time Step, Batch Size, 1, Sample Rate
        xt = self.time_step_replica(x)

        mems = {k: blk.init_leaky() for k, blk in self.conv_blks.named_children()}

        hist_y = []

        # Spikes of the intermediate layers
        # E.g. hist_spks['conv_blk1'][2] <- 3rd Time Step
        hist_spks = {k: [] for k, _ in self.conv_blks.named_children()}

        # Membrane of the intermediate layers
        hist_mems = {k: [] for k, _ in self.conv_blks.named_children()}

        # For each time step
        for step in range(self.n_steps):
            x = xt[step]

            for blk_name, blk in self.conv_blks.named_children():
                # blk_name: str. E.g. 'conv_blk1'
                x, mem = blk(x, mems[blk_name])

                # Append the spike and membrane history.
                hist_spks[blk_name].append(x.detach())
                hist_mems[blk_name].append(mem.detach())

                # Update the membrane potential.
                mems[blk_name] = mem

            x = self.avg_pool(x)
            y = self.classifier(x.permute(0, 2, 1))

            hist_y.append(y)

        yt = torch.stack(hist_y, dim=0)
        return yt, hist_mems, hist_spks

    @abstractmethod
    def time_step_replica(self, x) -> torch.Tensor:
        ...

    def step(self, batch):
        # x: Batch Size, 1, Sample Rate
        x, y_true = batch

        # yt: Time Step, Batch Size, 1, Classes
        yt, _, _ = self(x)

        # Sum it on the time step axes & squeeze
        # y: Batch Size, Classes
        y_pred_l = yt.sum(dim=0).squeeze().to(float)

        # Scale y to a proportion
        # We can consider moving the optimizer here
        # See: https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#snntorch.functional.loss.ce_count_loss
        # y_pred_l = y # / self.n_steps
        return x, y_pred_l, y_true
