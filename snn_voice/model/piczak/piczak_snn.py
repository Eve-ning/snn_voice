from collections import OrderedDict

from torch import nn

from snn_voice.model.piczak.blocks import PiczakSNNBlock


class PiczakSNN(nn.Module):
    def __init__(self, lif_beta: float, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ('snn',
                 nn.Sequential(
                     PiczakSNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), lif_beta, 0.5),
                     PiczakSNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), lif_beta, 0)
                 )),

                ('avg_pool', nn.AdaptiveAvgPool2d(1)),
                ('flatten', nn.Flatten(start_dim=1)),

                ('fc',
                 nn.Sequential(
                     nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU()),
                     nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU()),
                     nn.Sequential(nn.Linear(5000, n_classes))
                 ),
                 )
            ])
        )

    def forward(self, x):
        return self.net(x)

    # self.example_input_array = torch.rand([32, 1, 60, 101])
