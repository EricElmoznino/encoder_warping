import os
from pkgutil import get_data
import shutil

import hydra
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import r2_score
from models import ImageTransform, get_model
from omegaconf import DictConfig
from scripts.utils import get_datamodule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None, config_path="configurations", config_name="ols_linear_encoder"
)
def main(cfg: DictConfig) -> None:
    # Create save directory
    save_dir = f"saved_runs/ols_linear_encoder"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir)

    # Get model and datasets
    model, _, image_transform = get_model(
        cfg.model.arch, cfg.model.dataset, cfg.model.task, cfg.model.layer
    )
    model = model.to(device)
    datamodule = get_datamodule(cfg, image_transform, image_transform)
    datamodule.setup(stage=None)

    # Get the activations and labels
    x_train, x_val, x_test, y_train, y_val, y_test = get_data(model, datamodule)

    # Get the trained encoder
    encoder = ols_linear_encoder(x_train, y_train)

    # Get performance on all dataset splits
    with torch.no_grad():
        train_r2 = r2_score(encoder(x_train), y_train).item()
        val_r2 = r2_score(encoder(x_val), y_val).item()
        test_r2 = r2_score(encoder(x_test), y_test).item()

    # Save results
    results = pd.DataFrame(
        {"split": ["train", "val", "test"], "r2": [train_r2, val_r2, test_r2]}
    )
    results.to_csv(f"{save_dir}/results.csv", index=False)


def ols_linear_encoder(x: torch.Tensor, y: torch.Tensor) -> nn.Module:
    x = torch.concat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    w = torch.pinverse(x) @ y
    w, b = w[:-1], w[-1]

    encoder = nn.Linear(w.shape[0], w.shape[1])
    encoder.weight = nn.Parameter(w.T)
    encoder.bias = nn.Parameter(b)

    return encoder


def get_data(model: nn.Module, dm: pl.LightningDataModule) -> tuple[torch.Tensor, ...]:
    def get_activations_and_labels(
        dl: torch.utils.data.DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = [], []
        for x_batch, y_batch in tqdm(dl, desc="Getting activations"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                x_batch = model(x_batch)
            x.append(x_batch)
            y.append(y_batch)
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        return x, y

    x_train, y_train = get_activations_and_labels(dm.train_dataloader())
    x_val, y_val = get_activations_and_labels(dm.val_dataloader())
    x_test, y_test = get_activations_and_labels(dm.test_dataloader())

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    main()
