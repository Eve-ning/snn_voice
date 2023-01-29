from abc import ABC

from snn_voice.model.module import Module


class ModuleCNN(Module, ABC):
    """ Defines the base CNN Module """

    def step(self, batch):
        # y_true: (BS) in [0, N Classes)
        x, y_true = batch
        # y_pred_l: (BS, Classes Prob)
        y_pred_l = self(x).squeeze().to(float)
        return x, y_pred_l, y_true
