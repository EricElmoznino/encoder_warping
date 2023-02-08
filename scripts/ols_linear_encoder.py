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
    """
    Takes in a Hydra-specified configuration and trains a linear
    encoding model using either OLS or Ridge regression.

    Results are saved in `saved_runs/ols_linear_encoder/[run_name]`.
    `run_name` is dynamically generated based on the configuration.

    Args:
        cfg (DictConfig): A Hydra configuration object.
    """
    # Create save directory
    run_name = "_".join(
        [f"neural-data={cfg.data.name}_alpha={cfg.alpha}"]
        + [f"{k}={v}" for k, v in cfg.model.items()]
    )
    save_dir = f"saved_runs/ols_linear_encoder/{run_name}"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir)

    layerwise = False
    if cfg.layerwise == "layerwise":
        layerwise = True

    # Get model and datasets
    model, _, image_transform = get_model(
        cfg.model.arch, cfg.model.dataset, cfg.model.task, cfg.model.layer, layerwise
    )
    model = model.to(device)
    datamodule = get_datamodule(cfg, image_transform, image_transform)
    datamodule.setup(stage=None)

    # Get the activations and labels
    x_train, x_val, x_test, y_train, y_val, y_test = get_data(model, datamodule)

    # Get the trained encoder
    encoder = ols_linear_encoder(x_train, y_train, cfg.alpha)

    # Get performance on all dataset splits
    with torch.no_grad():
        train_r2 = r2_score(encoder(x_train), y_train).item()
        val_r2 = r2_score(encoder(x_val), y_val).item()
        test_r2 = r2_score(encoder(x_test), y_test).item()

    # Save results
    results = pd.DataFrame(
        {"split": ["train", "val", "test"], "r2": [train_r2, val_r2, test_r2]}
    )
    print(results)
    results.to_csv(f"{save_dir}/results.csv", index=False)


def ols_linear_encoder(
    x: torch.Tensor, y: torch.Tensor, alpha: float | None = None
) -> nn.Linear:
    """
    Fits an OLS or Ridge linear encoding model.

    Args:
        x (torch.Tensor): Regressors matrix.
        y (torch.Tensor): Targets matrix.
        alpha (float | None, optional): Ridge penalty. Defaults to None, in which case OLS is used.

    Returns:
        nn.Linear: A PyTorch linear model with the fitted OLS/Ridge parameters.
    """
    if alpha is None:
        alpha = 0.0

    x = torch.concat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    w = (
        torch.inverse(x.T @ x + alpha * torch.eye(x.shape[1], device=x.device))
        @ x.T
        @ y
    )
    w, b = w[:-1], w[-1]

    encoder = nn.Linear(w.shape[0], w.shape[1])
    encoder.weight = nn.Parameter(w.T)
    encoder.bias = nn.Parameter(b)

    return encoder


def get_data(model: nn.Module, dm: pl.LightningDataModule) -> tuple[torch.Tensor, ...]:
    """
    Makes a linear regression dataset based on the model's activations.

    Args:
        model (nn.Module): PyTorch model to use for obtaining activations.
        dm (pl.LightningDataModule): PyTorch Lightning datamodule containing stimuli and targets.

    Returns:
        tuple[torch.Tensor, ...]: Regressors and targets for training/validation/test splits.
    """

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
