import pytorch_lightning as pl

from src.datamodule.speech_command_datamodule import SpeechCommandDataModule
from src.model.cnn_piczak import CnnPiczak
from src.model.lit_wrapper import LitWrapper
from tests.evaluate import evaluate_model


def test_cnn_piczak():
    dm = SpeechCommandDataModule()  # dl_kwargs={'pin_memory': True}, num_workers=3)
    model = LitWrapper(CnnPiczak(len(dm.classes), n_channel=20), dm.classes, lr=0.01)
    # logger = TensorBoardLogger("test", log_graph=True)
    trainer = pl.Trainer(
        default_root_dir="cnn_piczak/",
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=1,
        limit_test_batches=1,
        accelerator='gpu',
        # logger=logger
        # fast_dev_run=True
    )

    evaluate_model(trainer, model, dm)

    # pred = trainer.predict(model, dataloaders=test_dl)
