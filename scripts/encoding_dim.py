import os
import shutil

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from models import get_model
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tasks import EncoderLinearTask, EncoderWarpingTask
from scripts.utils import get_datamodule


@hydra.main(config_path="configurations", config_name="encoding_dim")
def main(cfg: DictConfig) -> None:
    # Create save directory
    save_dir = f"saved_runs/encoding_dim/{cfg.run_name}"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir)

    # Sweep through the low dimensionality range
    low_dims = np.logspace(1, np.log10(cfg.dims.max), cfg.dims.n_samples - 1)
    low_dims = np.unique(low_dims.astype(int))
    low_dims = np.concatenate([[0], low_dims])

    results = []
    for low_dim in low_dims:
        test_r2 = train(cfg, save_dir, low_dim)
        results.append({"test_r2": test_r2, "low_dim": low_dim})

    results = pd.DataFrame(results)
    results.to_csv(f"{save_dir}/results.csv", index=False)


def train(cfg: DictConfig, save_dir: str, low_dim: int) -> float:
    save_dir = f"{save_dir}/low_dim={low_dim}"

    model, layer_groups, image_transform = get_model(cfg.model.arch, cfg.model.layer)
    datamodule = get_datamodule(
        cfg, image_transform, image_transform
    )  # TODO: Support training image transforms for data augmentation
    datamodule.setup(stage=None)

    if low_dim == 0:
        task = EncoderLinearTask(
            model=model,
            representation_size=model.representation_size,
            output_size=datamodule.n_outputs,
            lr=cfg.train.lr,
        )
    else:
        task = EncoderWarpingTask(
            model=model,
            low_dim=low_dim,
            layer_groups=layer_groups,
            representation_size=model.representation_size,
            output_size=datamodule.n_outputs,
            lr=cfg.train.lr,
        )

    # Configure trainer
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=cfg.train.max_epochs,
        callbacks=[
            ModelCheckpoint(monitor="val_r2", mode="max"),
            EarlyStopping(monitor="val_r2", mode="max"),
        ],
    )

    # Train and evaluate
    trainer.fit(task, datamodule=datamodule)
    test_r2 = trainer.test(datamodule=datamodule, ckpt_path="best")[0]["test_r2"]

    # Clear the model checkpoints to save disk space
    shutil.rmtree(f"{save_dir}/checkpoints", ignore_errors=True)

    return test_r2


if __name__ == "__main__":
    main()
