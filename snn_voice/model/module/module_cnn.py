from abc import ABC

from snn_voice.model.module import Module


class ModuleCNN(Module, ABC):
    """ Defines the base CNN Module """

    def forward(self, x):
        """ Forward-prop of x, yielding y_pred

        Notes:
            This will run through the CNN, the avg_pool, then the FCN
        """

        # TODO: should we change `conv_blks` to feature_extraction? Or another name?
        #  This is because we index the layers like so: net.conv_blks.conv_blk1, which may look redundant
        #  Could we consider net.feature_ext.conv_blk1?
        #  or net.cnn.blk1 (seems to be the best)
        #  or net.cnn.0 <- Does this work for unnamed modules? Will it break code that used .named_modules()?

        x = self.conv_blks(x)
        x = self.avg_pool(x).permute(0, 2, 1)
        return self.classifier(x)

    def step(self, batch):
        # y_true: (BS) in [0, N Classes)
        x, y_true = batch
        # y_pred_l: (BS, Classes Prob)
        y_pred_l = self(x).squeeze().to(float)
        return x, y_pred_l, y_true
