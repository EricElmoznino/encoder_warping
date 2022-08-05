from typing import Any, Literal

import pytorch_lightning as pl
import torch
from low_dim import FastfoodWrapper
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import Accuracy
from models import LayerGroups

Phase = Literal["train", "val", "test"]


class ClassifierLinearTask(pl.LightningModule):
    """
    PyTorch LightningModule representing a classification task.
    """

    def __init__(
        self,
        model: nn.Module,
        representation_size: int,
        n_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Initialize the task.

        Args:
            model (nn.Module): Candidate brain model (usually a deep neural network).
            representation_size (int): Number of outputs from the model.
            n_classes (int): Number of classes.
            lr (float, optional): Gradient descent. Defaults to 1e-4.
            weight_decay (float, optional): L2 weight regularization strength. Defaults to 0.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "representation_size", "n_classes"])

        self.model = model.eval()
        self.output_head = nn.Linear(representation_size, n_classes)

        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()

    def forward(self, x: Any) -> torch.Tensor:
        """
        Obtain the model's activations (without gradients)
        and linearly project them to the output space.

        Args:
            x (Any): Input(s) to self.model.

        Returns:
            torch.Tensor: Predicted classes.
        """
        with torch.no_grad():
            x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_head(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y_true)

        self.log("train_loss", loss)
        self.update_metrics(y_pred, y_true, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        self.update_metrics(y_pred, y_true, "val")

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        self.update_metrics(y_pred, y_true, "test")

    def training_epoch_end(self, outputs):
        self.log_metrics("train")

    def validation_epoch_end(self, outputs):
        self.log_metrics("val")

    def test_epoch_end(self, outputs):
        self.log_metrics("test")

    def update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, phase: Phase):
        """
        Call whenever you want to incrementally update the
        state of the internal metrics tracking performance.

        Args:
            y_pred (torch.Tensor): Predicted classes.
            y_true (torch.Tensor): True classes.
            phase (Phase): One of "train", "val", or "test".
        """
        acc = self.metrics_for_phase(phase)
        acc.update(y_pred, y_true)

    def log_metrics(self, phase: Phase):
        """
        Call whenever you want to compute and log the
        the current internal metrics tracking performance.

        Args:
            phase (Phase): One of "train", "val", or "test".
        """
        acc = self.metrics_for_phase(phase)
        self.log(f"{phase}_accuracy", acc.compute(), prog_bar=True)
        acc.reset()

    def metrics_for_phase(self, phase: Phase) -> Accuracy:
        """
        Convenience function to get the Accuracy object
        based on the current phase of training.

        Args:
            phase (Phase): One of "train", "val", or "test".

        Returns:
            Accuracy: Accuracy object for the current phase of training.
        """
        if phase == "train":
            return self.acc_train
        elif phase == "val":
            return self.acc_val
        else:
            return self.acc_test

    def configure_optimizers(self):
        return optim.Adam(
            self.output_head.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class ClassifierWarpingTask(ClassifierLinearTask):
    """
    PyTorch LightningModule representing a classification task
    used to finetune a nonlinear model, where
    the degree of nonlinear warping can be modulated by
    varying the dimensionality of the parameter embedding
    vector being trained.

    See :class:`low_dim.FastfoodWrapper` for more details
    on this nonlinear warping process.
    """

    def __init__(
        self,
        *args: list,
        low_dim: int,
        layer_groups: LayerGroups | None = None,
        low_dim_lr: float | None = None,
        **kwargs: dict,
    ) -> None:
        """
        Initialize the task

        Args:
            *args (list): Positional arguments passed to superclass. See :class:`ClassificationLinearTask`.
            low_dim (int): Dimensionality of the low-dimensional parameter embedding vector being trained.
            layer_groups (LayerGroups | None, optional): Layer groups that receive their own scaling parameters. Defaults to None.
            low_dim_lr (float | None, optional): Separate learning rate for low-dimensional parameter embedding vector. Defaults to None, in which case the same learning rate is used as for the linear output head (self.hparams.lr).
            **kwargs (dict): Keyword arguments passed to superclass. See :class:`ClassificationLinearTask`.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model", "representation_size", "n_classes"])

        self.model = FastfoodWrapper(
            model=self.model, low_dim=low_dim, layer_groups=layer_groups
        )

        # Disable automatic PyTorch Lightning's automatic optimization so that we can efficiently
        # use the model and head optimizers separately in the same call to self.training_step().
        self.automatic_optimization = False

    def forward(self, x):
        """
        Obtain the model's activations and linearly project them to the output space.

        Args:
            x (Any): Input(s) to self.model.

        Returns:
            torch.Tensor: Predicted classes.
        """
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_head(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss)
        for optimizer in optimizers:
            optimizer.step()

        return loss

    def configure_optimizers(self):
        if self.hparams.low_dim_lr is None:
            low_dim_lr = self.hparams.lr
        else:
            low_dim_lr = self.hparams.low_dim_lr
        model_optimizer = optim.Adam(self.model.parameters(), lr=low_dim_lr)

        head_optimizer = optim.Adam(
            self.output_head.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return [model_optimizer, head_optimizer]
