from abc import abstractmethod, ABC
from typing import Tuple

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim.lr_scheduler import StepLR

from src.settings import SPEECHCOMMAND_CLASSES, TOPK, LEARNING_RATE


class MxCommon(pl.LightningModule, ABC):
    """ Defines an abstract class that all M Models can inherit from """

    def __init__(
            self,
            lr: float = LEARNING_RATE,
            topk: Tuple[int] = TOPK,
            n_classes: int = len(SPEECHCOMMAND_CLASSES)
    ):
        super().__init__()
        self.lr = lr
        self.topk = topk
        self.n_classes = n_classes

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.criterion = nn.CrossEntropyLoss()
        # We'll set a static example, exposing the correct sizes is too much of a workaround
        self.example_input_array = torch.rand([32, 1, 4000])

        self.classifier = nn.Sequential()
        self.conv_blks = nn.Sequential()

    def training_step(self, batch, batch_ix):
        x, y_pred_l, y_true = self.step(batch)
        loss = self.criterion(y_pred_l, y_true)

        self.log('loss', loss)
        self.log_topk(y_pred_l, y_true, self.topk)

        return loss

    def validation_step(self, batch, batch_ix):
        x, y_pred_l, y_true = self.step(batch)
        loss = self.criterion(y_pred_l, y_true)

        self.log('val_loss', loss)
        self.log_topk(y_pred_l, y_true, self.topk, validation=True)

        return loss

    def predict_step(self, batch, batch_ix):
        """ Performs a batch prediction

        Returns:
            The batch input, predicted label, true label as a 3-tuple
        """
        x, y_pred_l, y_true = self.step(batch)
        y_pred = torch.argmax(y_pred_l, dim=1)
        return x, y_pred, y_true

    def log_topk(self, y_pred_l, y_true, ks: Tuple[int] = (1, 2, 5),
                 validation: bool = False):
        """ Logs the Top-K accuracies

        Args:
            ks: A Tuple of Top-K to evaluate.
            validation: Whether to log validation top-k
        """
        _, y_pred_topmax = torch.topk(y_pred_l, max(ks), dim=1)
        y_pred_topmax = y_pred_topmax.T
        for k in ks:
            y_pred_topk = y_pred_topmax[:k]
            y_matches = (y_pred_topk == y_true).sum(dim=0).to(float)
            acc_topk = y_matches.mean()
            if validation:
                self.log(f'Val Top{k} Acc', acc_topk)
            else:
                self.log(f'Top{k} Acc', acc_topk, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.0001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": StepLR(
                    optimizer, 1, gamma=0.2, verbose=True
                ),
                # "scheduler": ReduceLROnPlateau(
                # optimizer, 'min', patience=1, verbose=True
                # ),
                # "monitor": "val_loss",
                # },
            }
        }

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def step(self, batch):
        ...
