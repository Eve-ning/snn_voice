import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import Adam


class LitWrapper(pl.LightningModule):

    def __init__(self, model, classes, lr=0.005):
        super(LitWrapper, self).__init__()
        self.model = model
        self.classes = classes
        self.lr = lr

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
        return Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)
