import os
from pkgutil import get_data
import shutil

import hydra
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import r2_score
from models import ImageTransform, get_model
from omegaconf import DictConfig
from scripts.utils import get_datamodule
from scripts.regression import LinearRegression, regression_cv_concatenated
from datasets.nsd import get_nsd_dataset
from datasets.majaj import get_majaj_dataset
from torch.utils.data import DataLoader
from sklearn import linear_model

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
    if cfg.data.name == "nsd":
        dataset, _ = get_nsd_dataset(
                    cfg.data.nsd_dir,
                    cfg.data.stimuli_dir,
                    cfg.model.arch,
                    split="all",
                    image_transform=image_transform,
                )

    if cfg.data.name == "majaj":
        dataset, _ = get_majaj_dataset(
                    cfg.data.data_path,
                    cfg.data.stimuli_dir,
                    cfg.data.roi,
                    cfg.model.arch, 
                    split="all",
                    image_transform=image_transform,
        )

    features, responses = get_activations_and_labels(model, dataset)

    # Get performance on all dataset splits
    with torch.no_grad():
        regression_model = linear_model.LinearRegression()
        y_true, y_predicted = regression_cv_concatenated(x=features,y=responses,model=regression_model)
        r = pearson_r(y_true,y_predicted)

    print("Average R Value:")
    print(torch.mean(r))
    # Save results
    results = pd.DataFrame(
        {"R value": [r]}
    )

    results.to_csv(f"{save_dir}/results.csv", index=False)

def get_activations_and_labels(
        model, dataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dl = DataLoader(dataset, batch_size = 128)
        x, y = [], []
        for x_batch, y_batch in tqdm(dl, desc="Getting activations"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                x_batch = model(x_batch)
            x.append(x_batch)
            y.append(y_batch)
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        return x.cpu(), y.cpu()

def z_score(
    x: torch.Tensor, *, dim: int = 0, unbiased: bool = True, nan_policy: str = "omit"
) -> torch.Tensor:
    match nan_policy:
        case "propagate":
            x_mean = x.mean(dim=dim, keepdim=True)
            x_std = x.std(dim=dim, keepdim=True, unbiased=unbiased)
        case "omit":
            x_mean = x.nanmean(dim=dim, keepdim=True)
            ddof = 1 if unbiased else 0
            x_std = (
                ((x - x_mean) ** 2).sum(dim=dim, keepdim=True) / (x.shape[dim] - ddof)
            ).sqrt()
        case _:
            raise ValueError("x contains NaNs")

    x = (x - x_mean) / x_std
    return x


def _helper(
    x: torch.Tensor,
    y: torch.Tensor = None,
    *,
    return_value: str,
    return_diagonal: bool = True,
    unbiased: bool = True,
    nan_policy: str = "omit",
) -> torch.Tensor:
    if x.ndim not in {1, 2, 3}:
        raise ValueError(f"x must have 1, 2 or 3 dimensions (n_dim = {x.ndim})")
    x = x.unsqueeze(1) if x.ndim == 1 else x

    dim_sample_x, dim_feature_x = x.ndim - 2, x.ndim - 1
    n_samples_x = x.shape[dim_sample_x]
    n_features_x = x.shape[dim_feature_x]

    if return_value == "pearson_r":
        x = z_score(x, dim=dim_sample_x, unbiased=unbiased, nan_policy=nan_policy)
    elif return_value == "covariance":
        x = center(x, dim=dim_sample_x, nan_policy=nan_policy)

    if y is not None:
        if y.ndim not in {1, 2, 3}:
            raise ValueError(f"y must have 1, 2 or 3 dimensions (n_dim = {y.ndim})")
        y = y.unsqueeze(1) if y.ndim == 1 else y

        dim_sample_y, dim_feature_y = y.ndim - 2, y.ndim - 1
        n_samples_y = y.shape[dim_sample_y]

        if n_samples_x != n_samples_y:
            raise ValueError(
                f"x and y must have same n_samples (x={n_samples_x}, y={n_samples_y}"
            )

        if return_diagonal:
            n_features_y = y.shape[dim_feature_y]
            if n_features_x != n_features_y:
                raise ValueError(
                    "x and y must have same n_features to return diagonal"
                    f" (x={n_features_x}, y={n_features_y})"
                )

        if return_value == "pearson_r":
            y = z_score(y, dim=dim_sample_y, unbiased=unbiased, nan_policy=nan_policy)
        elif return_value == "covariance":
            y = center(y, dim=dim_sample_y, nan_policy=nan_policy)
    else:
        y = x

    x = torch.matmul(x.transpose(-2, -1), y) / (n_samples_x - 1)
    if return_diagonal:
        x = torch.diagonal(x, dim1=-2, dim2=-1)
    return x.squeeze()
    
def pearson_r(
    x: torch.Tensor,
    y: torch.Tensor = None,
    *,
    return_diagonal: bool = True,
    unbiased: bool = True,
    nan_policy: str = "omit",
) -> torch.Tensor:
    """Computes Pearson correlation coefficients.
    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise correlations are broadcasted along the batch dimension). If x and y are both specified, pairwise correlations 
    between the columns of x and those of y are computed.
    :param x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
    :param y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
    :param return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise correlation 
    matrix, defaults to True
    :return: Pearson correlation coefficients (*, n_features_x, n_features_y)
    """
    return _helper(
        x=x,
        y=y,
        return_value="pearson_r",
        return_diagonal=return_diagonal,
        unbiased=unbiased,
        nan_policy=nan_policy,
    )

if __name__ == "__main__":
    main()
