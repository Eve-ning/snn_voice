from pytorch_lightning import Trainer

from snn_voice.datamodule.speech_command_datamodule import SpeechCommandsDataModule
from snn_voice.model.mx.m5_cnn import M5CNN
from snn_voice.model.mx.m5_trsafs import M5TRSAFS

dm = SpeechCommandsDataModule(batch_size=16)
# dm.prepare_data()
# dm.setup()
model_cnn = M5CNN()
# model_snn = M5TRSAFS()
#
trainer = Trainer(
    # fast_dev_run=True,
    accelerator='gpu',
    limit_train_batches=128,
    limit_val_batches=16,
    limit_predict_batches=4,
    max_epochs=2
)
trainer.fit(model_cnn, datamodule=dm)
preds = trainer.predict(model_cnn, datamodule=dm)
# %%
# trainer.fit(model_snn, datamodule=dm)
# trainer.predict(model_snn, datamodule=dm)
