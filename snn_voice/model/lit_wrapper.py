import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitWrapper(pl.LightningModule):

    def __init__(self, model, classes,
                 lr=0.005,
                 weight_decay=0.0001):
        super(LitWrapper, self).__init__()
        self.model = model
        self.classes = classes
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, b, b_ix):
        x, y_true = b
        y_pred = self.model(x)
        loss = cross_entropy(y_pred, y_true)
        self.log('train_loss', loss)
        self.log('train_acc', self.acc(y_pred, y_true))
        return loss

    def validation_step(self, b, b_ix):
        x, y_true = b
        y_pred = self.model(x)
        loss = cross_entropy(y_pred, y_true)
        self.log('val_loss', loss)
        self.log('val_acc', self.acc(y_pred, y_true))

    def acc(self, y_pred, y_true):
        return (y_pred.argmax(dim=1) == y_true).sum() / y_true.shape[0]

    def predict_step(self, b, b_ix):
        x, y_true = b
        y_pred = self.model(x)
        return [self.classes[ix] for ix in y_pred.argmax(dim=1)]

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=2, verbose=True),
                "monitor": "val_acc",
            },
        }
