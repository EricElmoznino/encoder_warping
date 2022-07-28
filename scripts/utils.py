from datasets import BaseDataModule, NSDDataModule
from models import ImageTransform
from omegaconf import DictConfig


def get_datamodule(
    cfg: DictConfig, train_transform: ImageTransform, eval_transform: ImageTransform
) -> BaseDataModule:
    if cfg.data.name == "nsd":
        datamodule = NSDDataModule(
            nsd_dir=cfg.data.nsd_dir,
            stimuli_dir=cfg.data.stimuli_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.name}")
    return datamodule
