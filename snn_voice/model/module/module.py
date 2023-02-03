from abc import abstractmethod, ABC
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from snn_voice.settings import TOPK, LEARNING_RATE


class Module(pl.LightningModule, ABC):
    """ Defines an abstract class that all Models can inherit from """

    def __init__(
            self,
            lr: float = LEARNING_RATE,
    ):
        """ Initializes an abstract datamodule for inheritance

        Args:
            lr: Learning Rate
        """

        super().__init__()
        self.lr = lr
        self.topk = TOPK
        self.criterion = nn.CrossEntropyLoss()
        # We'll set a static example, exposing the correct sizes is too much of a workaround
        # self.example_input_array = torch.rand([32, 1, 4000])

    def training_step(self, batch, batch_ix):
        """ A single step for training

        Args:
            batch: (x, y_pred_l (continuous), y_true (label))
            batch_ix: Batch Index

        Returns:
            The loss based on criterion
        """
        x, y_pred_l, y_true = self.step(batch)
        loss = self.criterion(y_pred_l, y_true)

        self.log('loss', loss)
        self.log_topk(y_pred_l, y_true, self.topk)

        return loss

    def validation_step(self, batch, batch_ix):
        """ A single step for validation

        Notes:
            See self.training_step()
        """
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
                self.log(f'Val_Top{k}_Acc', acc_topk)
            else:
                self.log(f'Top{k}_Acc', acc_topk, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.0001
        )

        # Retrieves the appropriate steps_per_peoch for LR Scheduler in next line
        steps_per_epoch = self.trainer.limit_train_batches \
            if self.trainer.limit_train_batches > 1.0 \
            else len(self.trainer.datamodule.train_dataloader())

        # LR Scheduler through Cosine Annealing
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=0.2,
            three_phase=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }

    @abstractmethod
    def forward(self, x):
        """ Forward-prop of x, yielding y_pred """
        ...

    @abstractmethod
    def step(self, batch):
        """ A step, yielding the x, y_pred_l (continuous), and the y_true.
        To be fed into CrossEntropyLoss()(input=y_pred_l, target=y_true)

        Notes:
            Thus, y_pred_l must be continuous logits of shape (BS, C), then y_true is the indices of the true label.
            For example,
                y_pred_l = [[-0.1, 0.0, 0.1], [0.1, 0.0, -0.1]] (BS=2 ,C=3)
                y_true = [2, 1] (BS=2) in [0, 3)
                of batch size = 2.

            Will find the 1st element of the mini-batch correct, and the 2nd wrong
        """
        ...
