import pytorch_lightning as pl

from src.dataset.speech_command_dataset import SpeechCommandDataset
from src.model.cnn_piczak import CnnPiczak
from src.model.lit_wrapper import LitWrapper
from tests.evaluate import evaluate_model


def test_cnn_piczak():
    ds = SpeechCommandDataset(dl_kwargs={'pin_memory': True}, num_workers=3)
    model = LitWrapper(CnnPiczak(len(ds.classes), n_channel=20), ds.classes, lr=0.01)

    trainer = pl.Trainer(
        default_root_dir="cnn_piczak/",
        max_epochs=1,
        accelerator='gpu'
        # fast_dev_run=True
    )
    evaluate_model(trainer, model, ds)

    # pred = trainer.predict(model, dataloaders=test_dl)
