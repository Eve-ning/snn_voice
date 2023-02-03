from typing import Callable, List

import torch
from torch import nn

from snn_voice.model.module import ModuleSNN
from snn_voice.model.piczak.blocks import PiczakSNNBlock
from snn_voice.settings import DEFAULT_BETA


class PiczakSNN(ModuleSNN, nn.Module):
    def __init__(self, n_classes: int, n_steps: int,
                 time_step_replica: Callable[[torch.Tensor, int], torch.Tensor],
                 learn_beta: bool = True,
                 learn_thres: bool = True,
                 beta: float = DEFAULT_BETA,
                 *args, **kwargs):
        super().__init__(n_steps=n_steps, time_step_replica=time_step_replica, *args, **kwargs)
        self.snn = nn.ModuleList([
            PiczakSNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), beta, 0.5,
                           learn_beta=learn_beta, learn_thres=learn_thres),
            PiczakSNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), beta, 0,
                           learn_beta=learn_beta, learn_thres=learn_thres)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU()),
            nn.Sequential(nn.Linear(5000, n_classes))
        )
        self.time_step_replica_ = time_step_replica

    def time_step_forward(self, x, mems: List[torch.Tensor]):
        for e, blk in enumerate(self.snn):
            x, mems[e] = blk(x, mems[e])
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def init_mems(self) -> List[torch.Tensor]:
        return [blk.init_leaky() for blk in self.snn]
