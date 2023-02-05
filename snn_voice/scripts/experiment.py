import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from snn_voice.datamodule import SpeechCommandsDataModule, SampleDataModule  # noqa
from snn_voice.model.hjh import HjhCNN, HjhSCNN, HjhSNN  # noqa
from snn_voice.model.mx.m5 import M5CNN, M5SNN  # noqa
from snn_voice.model.piczak import PiczakCNN, PiczakSNN  # noqa
from snn_voice.model.tcy import TcyNN, TcySNN  # noqa
from snn_voice.scripts.config_schema import ConfigSchema
from snn_voice.utils.time_step_replica import repeat_replica, rate_replica, latency_replica  # noqa


def sanitize(x: str):
    return "".join(c for c in x if c.isalnum())


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def experiment(cfg: ConfigSchema) -> None:
    Model = eval(sanitize((cfg_m := cfg.model).name))
    model = Model(
        n_classes=cfg_m.data.n_classes,
        n_steps=cfg_m.n_steps,
        time_step_replica=eval(sanitize(cfg_m.time_step_replica) + "_replica"),
        learn_beta=cfg_m.learn_beta,
        learn_thres=cfg_m.learn_thres,
        beta=cfg_m.beta,
    )
    DataModule = eval(sanitize((cfg_d := cfg_m.data).name))
    dm = DataModule(
        n_mels=cfg_d.n_mels,
        batch_size=cfg_d.batch_size
    )
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        limit_train_batches=(cfg_t := cfg.trainer).limit_train_batches,
        limit_val_batches=cfg_t.limit_val_batches,
        max_epochs=cfg_t.max_epochs,
        fast_dev_run=cfg_t.fast_dev_run,
        callbacks=[
            LearningRateMonitor(),
            EarlyStopping(
                monitor=(cfg_tce := cfg_t.callbacks.early_stopping).monitor,
                patience=cfg_tce.patience,
                mode=cfg_tce.mode,
            )
        ]
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    experiment()
