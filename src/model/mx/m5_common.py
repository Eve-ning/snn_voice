from dataclasses import dataclass
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import StepLR


@dataclass
class M5Common(pl.LightningModule):
    le: LabelEncoder
    conv_blks: nn.Module
    lr: float = 0.01
    topk: Tuple[int] = (TOPK := (1, 2, 5))

    classifier: nn.Module = nn.Linear(512, (N_CLASSES := 35))
    avg_pool = nn.AdaptiveAvgPool1d(1)
    criterion = nn.CrossEntropyLoss()

    def __post_init__(self):
        """ Defines an abstract class that all M Models can inherit from """
        super().__init__()
        self.example_input_array = torch.rand([32, 1, NEW_SAMPLE_RATE])

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.avg_pool(x).permute(0, 2, 1)
        x = self.classifier(x)

        return x

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
        x, y_pred_l, y_true = self.step(batch)
        y_pred = torch.argmax(y_pred_l, dim=1)
        y_pred_lab = self.le.inverse_transform(y_pred)
        y_true_lab = self.le.inverse_transform(y_true)

        return x, y_pred_lab, y_true_lab

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
