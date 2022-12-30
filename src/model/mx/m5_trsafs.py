from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from torch import nn

from src.model.mx.m5_common import M5Common
from src.model.mx.m5_snn_block import M5SNNBlock


@dataclass
class M5TRSAFS(M5Common):
    n_steps: int = 2
    lif_beta: float = 0.2
    conv_blks: nn.Module = field(init=False, default_factory=lambda: nn.Sequential(
        OrderedDict([
            ('conv_blk1', M5SNNBlock(1, 128, 80, lif_beta, 4)),
            ('conv_blk2', M5SNNBlock(128, 128, 3, lif_beta)),
            ('conv_blk3', M5SNNBlock(128, 256, 3, lif_beta)),
            ('conv_blk4', M5SNNBlock(256, 512, 3, lif_beta)),
        ])
    ))

    def __post_init__(self):
        super().__init__()

    def forward(self, xt):
        """ We should expect a T x B x t input"""

        mems = [blk.init_leaky() for blk in self.conv_blks.children()]

        hist_y = []
        hist_spks = []
        hist_mems = []

        for step in range(self.n_steps):
            x = xt[step]

            for e, (mem, blk) in enumerate(zip(mems, self.conv_blks.children())):
                x, mem = blk(x, mem)
                hist_mems.append(mem.detach())
                hist_spks.append(x.detach())
                mems[e] = mem

            x = self.avg_pool(x)
            y = self.classifier(x.permute(0, 2, 1))

            hist_y.append(y)

        yt = torch.stack(hist_y, dim=0)
        return yt, hist_mems, hist_spks

    def step(self, batch):
        # x: Batch Size, 1, Sample Rate
        x, y_true = batch

        # xt: Time Step, Batch Size, 1, Sample Rate
        xt = x.repeat(self.n_steps, 1, 1, 1)

        # yt: Time Step, Batch Size, 1, Classes
        yt, _, _ = self(xt)

        # Sum it on the time step axes & squeeze
        # y: Batch Size, Classes
        y_pred_l = yt.sum(dim=0).squeeze().to(float)

        # Scale y to a proportion
        # We can consider moving the optimizer here
        # See: https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#snntorch.functional.loss.ce_count_loss
        # y_pred_l = y # / self.n_steps
        return x, y_pred_l, y_true
