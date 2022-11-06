import torch


def evaluate_model(trainer, model, ds):
    """ Evaluates a model

    Args:
        trainer: pl.Trainer Instance
        model: Model instance
        ds: Dataset instance

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} Backend!")

    train_dl, val_dl, test_dl = ds.dls()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, )
