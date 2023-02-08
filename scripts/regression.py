from abc import ABC, abstractmethod
from collections.abc import Collection

from tqdm import tqdm
import numpy as np
import torch


class Regression(ABC):
    ...

    @abstractmethod
    def fit(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        ...

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...


class LinearRegression(Regression):
    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy: bool = True,
        l2_penalty: float | int | torch.Tensor = None,
        rcond: float = None,
        driver: str = None,
        allow_ols_on_cuda: bool = False,
    ) -> None:
        # TODO test for 3D x
        self.fit_intercept = fit_intercept
        if isinstance(l2_penalty, float | int):
            if l2_penalty == 0:
                self.l2_penalty = None
            else:
                self.l2_penalty = torch.tensor(l2_penalty)
        else:
            self.l2_penalty = l2_penalty
        self.device: torch.device = None

        self.copy: bool = copy
        self.rcond: float = rcond
        self.driver: str = driver
        self.allow_ols_on_cuda: bool = allow_ols_on_cuda

        self.n_samples: int = None
        self.n_features: int = None
        self.coefficients: torch.Tensor = None
        self.intercept: torch.Tensor = None
        self.rank: int = None
        self.residuals: torch.Tensor = None
        self.singular_values: torch.Tensor = None

    def fit(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self.device = x.device
        if self.copy:
            x = torch.clone(x)
            y = torch.clone(y).to(self.device)
        else:
            x = x
            y = y.to(self.device)

        # only 1 predictor
        x = x.unsqueeze(dim=-1) if x.ndim == 1 else x

        # only 1 target
        y = y.unsqueeze(dim=-1) if y.ndim == 1 else y

        # many sets of predictors, only 1 set of targets
        if x.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(0)

        self.n_samples, self.n_features = x.shape[-2], x.shape[-1]

        # TODO: underdetermined systems on CUDA use a different driver
        if (not self.allow_ols_on_cuda) and (self.l2_penalty is None):
            if self.n_samples < self.n_features:
                self.device = torch.device("cpu")
                x = x.to(self.device)
                y = y.to(self.device)

        if y.shape[-2] != self.n_samples:
            raise ValueError(
                f"number of samples in x and y must be equal (x={self.n_samples},"
                f" y={y.shape[-2]})"
            )

        if self.fit_intercept:
            x_mean = x.mean(dim=-2, keepdim=True)
            x = x - x_mean
            y_mean = y.mean(dim=-2, keepdim=True)
            y = y - y_mean

        if self.l2_penalty is None:  # OLS
            self._solve_lstsq(x=x, y=y)
        else:  # ridge
            self._solve_svd(x=x, y=y)

        if self.fit_intercept:
            self.intercept = y_mean - torch.matmul(x_mean, self.coefficients)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fit")
        product = torch.matmul(x.to(self.device), self.coefficients)
        if self.fit_intercept:
            return product + self.intercept
        else:
            return product

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.coefficients is not None:
            self.coefficients = self.coefficients.to(device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(device)
        if self.residuals is not None:
            self.residuals = self.residuals.to(device)
        if self.singular_values is not None:
            self.singular_values = self.singular_values.to(device)

    def _solve_lstsq(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        (
            self.coefficients,
            self.residuals,
            self.rank,
            self.singular_values,
        ) = torch.linalg.lstsq(x, y, rcond=self.rcond, driver=self.driver)

    def _solve_svd(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        l2_penalty = self.l2_penalty.to(self.device)
        if l2_penalty.numel() == 1:
            l2_penalty = l2_penalty * torch.ones(y.shape[-1], device=self.device)

        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        idx = s > 1e-15
        s_nnz = s[idx].unsqueeze(-1)
        d = torch.zeros(
            size=(len(s), l2_penalty.numel()), dtype=x.dtype, device=self.device
        )
        d[idx] = s_nnz / (s_nnz**2 + l2_penalty)
        self.coefficients = torch.matmul(
            vt.transpose(-2, -1), d * torch.matmul(u.transpose(-2, -1), y)
        )


def create_splits(
    n: int, *, n_folds: int = 5, shuffle: bool = True
) -> list[np.ndarray[int]]:
    if shuffle:
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)

    x = np.array_split(indices, n_folds)
    return x


def regression(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Regression = LinearRegression(),
    indices_train: Collection[int] = None,
    indices_test: Collection[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices_train is None and indices_test is not None:
        indices_train = np.setdiff1d(np.arange(x.shape[-2]), np.array(indices_test))
    elif indices_test is None and indices_train is not None:
        indices_test = np.setdiff1d(np.arange(x.shape[-2]), np.array(indices_train))
    elif indices_train is None and indices_test is None:
        indices_train = np.arange(x.shape[-2])
        indices_test = np.arange(x.shape[-2])

    x_train, x_test = x[..., indices_train, :], x[..., indices_test, :]
    y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]

    model.fit(X=x_train, y=y_train)

    y_predicted = model.predict(x_test)
    return y_test, y_predicted


def regression_cv(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Regression = LinearRegression(),
    n_folds: int = 8,
    shuffle: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    y_true, y_predicted = [], []

    splits = create_splits(n=y.shape[-2], n_folds=n_folds, shuffle=shuffle)
    for indices_test in tqdm(splits, desc="split", leave=False):
        y_true_, y_predicted_ = regression(
            model=model,
            x=x,
            y=y,
            indices_test=indices_test,
        )
        y_true.append(y_true_)
        y_predicted.append(y_predicted_)

    return y_true, y_predicted


def regression_cv_concatenated(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Regression = LinearRegression(),
    n_folds: int = 8,
    shuffle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_true, y_predicted = regression_cv(
        x=x, y=y, model=model, n_folds=n_folds, shuffle=shuffle
    )

    y_predicted = [torch.from_numpy(item) for item in y_predicted]
    y_predicted = torch.concat(y_predicted, dim=-2)
    y_true = torch.concat(y_true, dim=-2)
    return y_true, y_predicted