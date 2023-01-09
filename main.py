from snn_voice.datamodule import SpeechCommandsDataModule

dm_spec = SpeechCommandsDataModule(batch_size=16, n_mels=60)
