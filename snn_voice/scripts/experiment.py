from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from snn_voice.datamodule import SpeechCommandsDataModule, SampleDataModule  # noqa
from snn_voice.model.hjh import HjhCNN, HjhSCNN, HjhSNN  # noqa
from snn_voice.model.module import ModuleSNN
from snn_voice.model.mx.m5 import M5CNN, M5SNN  # noqa
from snn_voice.model.piczak import PiczakCNN, PiczakSNN  # noqa
from snn_voice.model.tcy import TcyNN, TcySNN  # noqa
from snn_voice.scripts.config_schema import ConfigSchema
from snn_voice.utils.time_step_replica import repeat_replica, rate_replica, latency_replica  # noqa

from hydra.core.hydra_config import HydraConfig


def sanitize(x: str):
    return "".join(c for c in x if c.isalnum())


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def experiment(cfg: ConfigSchema) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    print(f"Working Directory: {output_dir}")
    if cfg.test_config:
        return
    Model = eval(sanitize((cfg_m := cfg.model).name))

    if issubclass(Model, ModuleSNN):
        model = Model(
            n_classes=cfg_m.data.n_classes,
            n_steps=cfg_m.snn.n_steps,
            time_step_replica=eval(sanitize(cfg_m.snn.time_step_replica) + "_replica"),
            learn_beta=cfg_m.snn.learn_beta,
            learn_thres=cfg_m.snn.learn_thres,
            beta=cfg_m.snn.beta,
        )
    else:
        model = Model(n_classes=cfg_m.data.n_classes)
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
        ],
        default_root_dir=output_dir.as_posix()
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    experiment()
