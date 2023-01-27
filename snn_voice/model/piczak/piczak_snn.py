from collections import OrderedDict

from torch import nn

from snn_voice.model.piczak.blocks import PiczakSNNBlock


def piczak_snn_init(self):
    """ Initializes the Piczak __init__ blocks """
    self.net = nn.Sequential(
        OrderedDict([
            ('snn1', PiczakSNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), self.lif_beta, 0.5)),
            ('snn2', PiczakSNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), self.lif_beta, 0)),

            ('avg_pool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten(start_dim=1)),

            ('fc1', nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU())),
            ('fc2', nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU())),
            ('fc3', nn.Sequential(nn.Linear(5000, self.n_classes))),
        ])
    )

    # self.example_input_array = torch.rand([32, 1, 60, 101])
