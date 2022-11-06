import pytorch_lightning as pl

from src.dataset.speech_command_dataset import SpeechCommandDataset
from src.model.lit_wrapper import LitWrapper
from src.model.srnn_hjh import SrnnHJH
from tests.evaluate import evaluate_model


def test_srnn_hjh():
    ds = SpeechCommandDataset(batch_size=128, num_workers=3, dl_kwargs={'pin_memory': True})
    model = LitWrapper(SrnnHJH(
        len(ds.classes), lstm_n_layers=1, lstm_n_hidden=48, n_time=16, resample=(16000, 8000)
    ), ds.classes, lr=0.01).to('cuda')

    trainer = pl.Trainer(
        default_root_dir="srnn_hjh/",
        max_epochs=5,
        accelerator='gpu'
        # fast_dev_run=True
    )
    evaluate_model(trainer, model, ds)

    # pred = trainer.predict(model, dataloaders=test_dl)
