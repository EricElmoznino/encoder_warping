"""
Script that can be used to train toy models to fit a target function,
while varying the level of complexity through the dimensionality
of the model's trainable parameter embedding.

Can run using the following command:
`python -m simulations.dim_complexity.train_models`

There are no command-line arguments for this simple script.
Simply change the global variables indicated below.

The results can be analyzed  in the `simulations/dim_complexity/figures.ipynb` notebook.
"""

import copy
from typing import Callable

import numpy as np
import torch
from low_dim.fastfood import FastfoodWrapper
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

# Change the variables below to modify the simulation.
# target_function: the target function to fit.
# initial_functions: the starting point functions of the models.
# low_dims: the low dimensional parameter embedding sizes to try.
target_function = lambda x: torch.sin(1.5 * x * np.pi / 2)
initial_functions = [
    lambda x: x**2,
    lambda x: x**3,
    lambda x: torch.sin(x * np.pi / 2),
]
low_dims = np.logspace(np.log10(4), 2, 7).astype(int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(
    batch_size: int, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    x = torch.randn(batch_size, 1).to(device)
    with torch.no_grad():
        y = func(x)
    return x, y


def fit_to_func(
    model: nn.Module,
    func: Callable[[torch.Tensor], torch.Tensor],
    lr: float = 1e-4,
    batch_size: int = 1000,
    threshold: float = 1e-5,
    patience: int = 2000,
    max_steps: int = 100000,
) -> nn.Module:
    model = copy.deepcopy(model)
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, threshold=threshold * 10
    )

    pbar = tqdm(range(max_steps))
    lowest_loss, patience_clock = np.inf, 0
    for step in pbar:
        x, y = get_batch(batch_size, func)
        y_pred = model(x)
        loss = F.smooth_l1_loss(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        loss = loss.item()
        dynamic_threshold = lowest_loss * (1 - threshold)
        if loss > dynamic_threshold:
            patience_clock += 1
        else:
            patience_clock = 0
        if loss < lowest_loss:
            lowest_loss = loss
        if patience_clock >= patience:
            break

        if step % 100 == 0:
            pbar.set_description(f"Loss: {lowest_loss:.5f}")

    model.eval()

    return model


class SimModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False),
        )
        self.layer_groups = ["layers.0", "layers.2", "layers.4"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def get_models() -> tuple[list[nn.Module], list[list[nn.Module]]]:
    init_models = [fit_to_func(SimModel(), func) for func in initial_functions]

    low_dim_models = []
    for low_dim in low_dims:
        models = []
        for model in init_models:
            model = FastfoodWrapper(model, low_dim, model.layer_groups)
            model = fit_to_func(model, target_function)
            models.append(model)
        low_dim_models.append(models)

    return init_models, low_dim_models


def save_models(
    init_models: list[nn.Module],
    low_dim_models: list[list[nn.Module]],
) -> None:
    save_dir = "simulations/dim_complexity/saved_models"

    for i, model in enumerate(init_models):
        torch.save(model.state_dict(), f"{save_dir}/func={i}.pth")

    for i, (dim, models) in enumerate(zip(low_dims, low_dim_models)):
        for j, model in enumerate(models):
            torch.save(model.state_dict(), f"{save_dir}/dim={i},func={j}.pth")


def main():
    init_models, low_dim_models = get_models()
    save_models(init_models, low_dim_models)


if __name__ == "__main__":
    main()
