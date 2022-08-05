import os
from pkgutil import get_data
import shutil

import hydra
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from models import ImageTransform, get_model
from omegaconf import DictConfig
from scripts.utils import get_datamodule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None, config_path="configurations", config_name="logistic_regression"
)
def main(cfg: DictConfig) -> None:
    """
    Takes in a Hydra-specified configuration and trains a linear
    logistic regression model.

    Results are saved in `saved_runs/logistic_regression/[run_name]`.
    `run_name` is dynamically generated based on the configuration.

    Args:
        cfg (DictConfig): A Hydra configuration object.
    """
    # Create save directory
    run_name = "_".join(
        [f"class-data={cfg.data.name}"] + [f"{k}={v}" for k, v in cfg.model.items()]
    )
    save_dir = f"saved_runs/logistic_regression/{run_name}"
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

    # Get the trained classifier
    classifier = logistic_regression(x_train, y_train)

    # Get performance on all dataset splits
    with torch.no_grad():
        train_acc = accuracy(classifier(x_train), y_train).item()
        val_acc = accuracy(classifier(x_val), y_val).item()
        test_acc = accuracy(classifier(x_test), y_test).item()

    # Save results
    results = pd.DataFrame(
        {"split": ["train", "val", "test"], "accuracy": [train_acc, val_acc, test_acc]}
    )
    print(results)
    results.to_csv(f"{save_dir}/results.csv", index=False)


def logistic_regression(x: torch.Tensor, y: torch.LongTensor) -> nn.Linear:
    """
    Fits a logistic regression classifier.

    Args:
        x (torch.Tensor): Feature matrix.
        y (torch.Tensor): Targets vector.

    Returns:
        nn.Linear: A PyTorch linear model with the fitted logistic regression parameters.
    """
    reg = LogisticRegression()
    reg.fit(x.cpu().numpy(), y.cpu().numpy())
    w, b = reg.coef_, reg.intercept_
    w, b = (
        torch.from_numpy(w).float().to(x.device),
        torch.from_numpy(b).float().to(y.device),
    )

    classifier = nn.Linear(w.shape[1], w.shape[0])
    classifier.weight = nn.Parameter(w)
    classifier.bias = nn.Parameter(b)

    return classifier


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
