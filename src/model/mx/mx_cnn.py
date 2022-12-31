from abc import ABC

from src.model.mx.mx_common import MxCommon


class MxCNN(MxCommon, ABC):

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x).permute(0, 2, 1)
        x = self.classifier(x)

        return x

    def step(self, batch):
        x, y_true = batch
        y_pred_l = self(x).squeeze().to(float)
        return x, y_pred_l, y_true
