from pkgutil import get_data
import hydra
from omegaconf import DictConfig
from models import get_model, ImageTransform
from datasets import BaseDataModule, NSDDataModule


@hydra.main(version_base=None, config_path="configurations", config_name="encoding_dim")
def main(cfg: DictConfig) -> None:
    model, layer_groups, image_transform = get_model(cfg.model.arch, cfg.model.layer)
    datamodule = get_datamodule(
        cfg, image_transform, image_transform
    )  # TODO: Support training image transforms for data augmentation


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


if __name__ == "__main__":
    main()
