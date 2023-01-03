from abc import ABC

from snn_voice.model.module import Module


class ModuleCNN(Module, ABC):
    """ Defines the base CNN Module """

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x).permute(0, 2, 1)
        return self.classifier(x)

    def step(self, batch):
        x, y_true = batch
        y_pred_l = self(x).squeeze().to(float)
        return x, y_pred_l, y_true
