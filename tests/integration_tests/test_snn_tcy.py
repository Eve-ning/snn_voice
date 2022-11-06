import pytorch_lightning as pl

from src.dataset.speech_command_dataset import SpeechCommandDataset
from src.model.lit_wrapper import LitWrapper
from src.model.snn_tcy import SnnTCY
from tests.evaluate import evaluate_model


def test_snn_tcy():
    ds = SpeechCommandDataset(batch_size=32)
    model = LitWrapper(SnnTCY(len(ds.classes)), ds.classes, lr=0.01)

    trainer = pl.Trainer(
        default_root_dir="snn_tcy/",
        max_epochs=1,
        # fast_dev_run=True
    )
    evaluate_model(trainer, model, ds)

    # pred = trainer.predict(model, dataloaders=test_dl)
