from abc import ABC
from collections import OrderedDict

from torch import nn

from snn_voice.model.module.module_cnn import ModuleCNN
from snn_voice.model.piczak.piczak_cnn_block import PiczakCNNBlock


class PiczakCNN(ModuleCNN, ABC):

    def __init__(self, sr: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # # We'll figure out the FFT window needed to satisfy the minimum window ms
        # n_fft = int(MIN_WINDOW_MS / (1 / sr * 1000))
        #
        # self.mel_spec = MelSpectrogram(
        #     n_mels=60,
        #     n_fft=n_fft
        # )

        self.conv_blks = nn.Sequential(
            OrderedDict([
                ('conv_blk1', PiczakCNNBlock(1, 80, (57, 6), (1, 1), (4, 3), (1, 3), 0.5)),
                ('conv_blk2', PiczakCNNBlock(80, 80, (1, 3), (1, 1), (1, 3), (1, 3), None)),
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Sequential(nn.Linear(80, 5000), nn.Dropout(0.5), nn.ReLU())),
                ('fc2', nn.Sequential(nn.Linear(5000, 5000), nn.Dropout(0.5), nn.ReLU())),
                ('fc3', nn.Sequential(nn.Linear(5000, self.n_classes)))
            ])
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x).squeeze()
        x = self.classifier(x)

        return x


# %%
net = PiczakCNN(4000)
net(net.example_input_array).shape
