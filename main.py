from pytorch_lightning import Trainer

from src.datamodule.speech_command_datamodule import SpeechCommandsDataModule
from src.model.mx.m5_cnn import M5CNN
from src.model.mx.m5_trsafs import M5TRSAFS

dm = SpeechCommandsDataModule()

model_cnn = M5CNN(dm.le)
model_snn = M5TRSAFS(dm.le)

trainer = Trainer(fast_dev_run=True)
trainer.fit(model_cnn, datamodule=dm)
trainer.predict(model_cnn, datamodule=dm)

trainer.fit(model_snn, datamodule=dm)
trainer.predict(model_snn, datamodule=dm)
