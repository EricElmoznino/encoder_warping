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
from tasks import ClassifierLinearTask, ClassifierWarpingTask
from scripts.utils import get_datamodule


@hydra.main(
    version_base=None, config_path="configurations", config_name="classifier_dim"
)
def main(cfg: DictConfig) -> None:
    """
    Takes in a Hydra-specified configuration and trains a set
    of classification models through gradient descent with varying
    sizes for the low-dimensional parameter embedding vector.

    Results (TensorBoard logs, model checkpoints, and final test set performances)
    are saved in `saved_runs/classification_dim/[run name]`.
    If `run_name` is not specified in the configuration, it
    is automatically generated based on the other configuration arguments.

    Args:
        cfg (DictConfig): A Hydra configuration object.
    """
    # Create save directory
    if cfg.run_name is None:
        cfg.run_name = "_".join(
            [f"class-data={cfg.data.name}"] + [f"{k}={v}" for k, v in cfg.model.items()]
        )
    save_dir = f"saved_runs/classifier_dim/{cfg.run_name}"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir)

    # Sweep through the low dimensionality range
    low_dims = np.logspace(1, np.log10(cfg.dims.max), cfg.dims.n_samples - 1)
    low_dims = np.unique(low_dims.astype(int))
    low_dims = np.concatenate([[0], low_dims])

    # Train models for each low dimensionality value
    results = []
    for low_dim in low_dims:
        test_acc = train(cfg, save_dir, low_dim)
        results.append({"test_accuracy": test_acc, "low_dim": low_dim})

    # Save results
    results = pd.DataFrame(results)
    print(results)
    results.to_csv(f"{save_dir}/results.csv", index=False)


def train(cfg: DictConfig, save_dir: str, low_dim: int) -> float:
    """
    Train a model with a given low-dimensional parameter embedding vector size.

    Args:
        cfg (DictConfig): A Hydra configuration object.
        save_dir (str): Directory to save TensorBoard logs and model checkpoints.
        low_dim (int): Size of the low-dimensional parameter embedding vector.

    Returns:
        float: Performance on the test set.
    """
    save_dir = f"{save_dir}/low_dim={low_dim}"

    model, layer_groups, image_transform = get_model(
        cfg.model.arch, cfg.model.dataset, cfg.model.task, cfg.model.layer
    )
    datamodule = get_datamodule(
        cfg, image_transform, image_transform
    )  # TODO: Support training image transforms for data augmentation
    datamodule.setup(stage=None)

    if low_dim == 0:
        task = ClassifierLinearTask(
            model=model,
            representation_size=model.representation_size,
            n_classes=datamodule.n_outputs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        task = ClassifierWarpingTask(
            model=model,
            low_dim=low_dim,
            layer_groups=layer_groups,
            representation_size=model.representation_size,
            n_classes=datamodule.n_outputs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

    # Configure trainer
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=cfg.train.max_epochs,
        callbacks=[
            ModelCheckpoint(monitor="val_accuracy", mode="max"),
            EarlyStopping(monitor="val_accuracy", mode="max", min_delta=1e-4),
        ],
    )

    # Train and evaluate
    trainer.fit(task, datamodule=datamodule)
    test_acc = trainer.test(datamodule=datamodule, ckpt_path="best")[0]["test_accuracy"]

    # Clear the model checkpoints to save disk space
    shutil.rmtree(
        f"{save_dir}/lightning_logs/version_0/version0/checkpoints", ignore_errors=True
    )

    return test_acc


if __name__ == "__main__":
    main()
