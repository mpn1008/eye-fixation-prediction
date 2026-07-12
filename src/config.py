from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    train_root: Path = Path("data/training")
    val_root: Path = Path("data/validation")
    train_images_txt: str = "train_images.txt"
    train_fixations_txt: str = "train_fixations.txt"
    val_images_txt: str = "val_images.txt"
    val_fixations_txt: str = "val_fixations.txt"
    image_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    backbone: str = "resnet101"
    pretrained: bool = True
    decoder_channels: list = field(default_factory=lambda: [256, 128, 64, 32])


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    # loss weights (kl + cc + mse)
    w_kl: float = 1.0
    w_cc: float = 1.0
    w_mse: float = 0
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs")
    save_every: int = 5  # save checkpoint every N epochs


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
