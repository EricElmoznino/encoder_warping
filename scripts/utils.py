from datasets import BaseDataModule, NSDDataModule, MajajDataModule
from models import ImageTransform
from omegaconf import DictConfig


def get_datamodule(
    cfg: DictConfig, train_transform: ImageTransform, eval_transform: ImageTransform
) -> BaseDataModule:
    """
    Convenience function to grab the correct dataset based on a script's configuration.

    Args:
        cfg (DictConfig): Hydra configuration object describing desired dataset.
        train_transform (ImageTransform): Image transform to apply to training data.
        eval_transform (ImageTransform): Image transform to apply to evaluation data.

    Raises:
        ValueError: Unimplemented dataset specified by the configuration.

    Returns:
        BaseDataModule: A subclass of BaseDataModule that implements the correct dataset.
    """
    if cfg.data.name == "nsd":
        datamodule = NSDDataModule(
            nsd_dir=cfg.data.nsd_dir,
            stimuli_dir=cfg.data.stimuli_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )
    elif cfg.data.name == "majaj":
        datamodule = MajajDataModule(
            data_path=cfg.data.data_path,
            stimuli_dir=cfg.data.stimuli_dir,
            roi=cfg.data.roi,
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.name}")
    return datamodule
