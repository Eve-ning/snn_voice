from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    patience: int
    mode: str
    monitor: str


@dataclass
class Callbacks:
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)


@dataclass
class Trainer:
    max_epochs: int
    limit_train_batches: int
    limit_val_batches: int
    fast_dev_run: bool
    accelerator: str
    callbacks: Callbacks = field(default_factory=Callbacks)


@dataclass
class Data:
    name: str
    n_mels: int
    n_classes: int
    batch_size: int


@dataclass
class Model:
    name: str
    data: Data
    n_steps: int
    time_step_replica: str
    learn_beta: bool
    learn_thres: bool
    beta: float


@dataclass
class ConfigSchema:
    trainer: Trainer
    model: Model
