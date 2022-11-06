import pytest
import pytorch_lightning as pl
import torch

from src.dataset.sample_dataset import SampleDataset
from src.model.cnn_m5 import CnnM5
from src.model.cnn_piczak import CnnPiczak
from src.model.lit_wrapper import LitWrapper
from src.model.srnn_hjh import SrnnHJH
from src.model.snn_tcy import SnnTCY


@pytest.mark.parametrize(
    'Model',
    [CnnM5, CnnPiczak, SnnTCY, SrnnHJH]
)
@pytest.mark.parametrize(
    'Dataset',
    [SampleDataset, ]
)
def test_models(Model, Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} Backend!")

    ds = Dataset()
    train_dl, val_dl, test_dl = ds.dls()

    model = LitWrapper(Model(len(ds.classes)), ds.classes, lr=0.01)

    trainer = pl.Trainer(
        default_root_dir="/",
        max_epochs=2,
        fast_dev_run=True
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, )
    # pred = trainer.predict(model, dataloaders=test_dl)
