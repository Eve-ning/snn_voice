import pytorch_lightning as pl

from src.dataset.speech_command_dataset import SpeechCommandDataset
from src.model.cnn_m5 import CnnM5
from src.model.lit_wrapper import LitWrapper
from tests.evaluate import evaluate_model


def test_cnn_m5():
    ds = SpeechCommandDataset(dl_kwargs={'pin_memory': True}, num_workers=3)
    model = LitWrapper(CnnM5(len(ds.classes)), ds.classes, lr=0.01)

    trainer = pl.Trainer(
        default_root_dir="cnn_m5/",
        max_epochs=1,
        # fast_dev_run=True
    )
    evaluate_model(trainer, model, ds)

    # pred = trainer.predict(model, dataloaders=test_dl)
