import pytorch_lightning as pl
import torch


def evaluate_model(trainer: pl.Trainer,
                   model: pl.LightningModule,
                   dm: pl.LightningDataModule):
    """ Evaluates a model

    Args:
        trainer: pl.Trainer Instance
        model: Model instance
        dm: DataModule instance

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} Backend!")

    trainer.fit(model, datamodule=dm)
