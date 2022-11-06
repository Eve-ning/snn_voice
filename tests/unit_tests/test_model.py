import pytest
import pytorch_lightning as pl
import torch

from src.datamodule.sample_datamodule import SampleDataModule
from src.model.cnn_m5 import CnnM5
from src.model.cnn_piczak import CnnPiczak
from src.model.lit_wrapper import LitWrapper
from src.model.snn_tcy import SnnTCY
from src.model.srnn_hjh import SrnnHJH


@pytest.mark.parametrize(
    'Model',
    [CnnM5, CnnPiczak, SnnTCY, SrnnHJH]
)
@pytest.mark.parametrize(
    'DataModule',
    [SampleDataModule, ]
)
def test_models(Model, DataModule):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} Backend!")

    dm = DataModule()
    model = LitWrapper(Model(len(dm.classes)), dm.classes, lr=0.01)

    trainer = pl.Trainer(
        default_root_dir="/",
        max_epochs=2,
        fast_dev_run=True
    )

    trainer.fit(model, datamodule=dm)
    # pred = trainer.predict(model, dataloaders=test_dl)
