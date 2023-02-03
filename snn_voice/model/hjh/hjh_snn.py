from typing import Callable, List

import snntorch as snn
import torch
from torch import nn

from snn_voice.model.hjh.blocks import HjhSNNBlock
from snn_voice.model.module import ModuleSNN


class HjhSNN(ModuleSNN, nn.Module):
    def __init__(self, n_classes: int, n_steps: int,
                 time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
                 learn_beta: bool = True,
                 learn_thres: bool = True,
                 beta: float = 0.5,
                 *args, **kwargs):
        super().__init__(n_steps=n_steps, time_step_replica=time_step_replica, *args, **kwargs)
        self.snn = nn.ModuleList([
            snn.Leaky(beta=beta, learn_beta=learn_beta, learn_threshold=learn_thres),
            HjhSNNBlock(1, 8, 5, max_pool_ksize=2, beta=beta, learn_beta=learn_beta, learn_thres=learn_thres),
            HjhSNNBlock(8, 16, 5, beta=beta, learn_beta=learn_beta, learn_thres=learn_thres),
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16, n_classes)

    def time_step_forward(self, x, mems: List[torch.Tensor]) -> torch.Tensor:
        for e, blk in enumerate(self.snn):
            x, mems[e] = blk(x, mems[e])
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def init_mems(self) -> List[torch.Tensor]:
        return [blk.init_leaky() for blk in self.snn]
