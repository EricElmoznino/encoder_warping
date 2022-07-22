from typing import Literal

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import R2Score

from low_dim import FastfoodWrapper

Phase = Literal["train", "val", "test"]


class EncoderLinearTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        representation_size: int,
        output_size: int,
        lr: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "representation_size", "output_size"]
        )

        self.model = model.eval()
        self.output_head = nn.Linear(representation_size, output_size)

        self.r2_train = R2Score(num_outputs=output_size)
        self.r2_val = R2Score(num_outputs=output_size)
        self.r2_test = R2Score(num_outputs=output_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        loss = F.mse_loss(y_pred, y_true)

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
        self.log_metrics()

    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def test_on_epoch_end(self, outputs):
        self.log_metrics()

    def update_metrics(self, y_pred, y_true, phase: Phase):
        r2 = self.metrics_for_phase(phase)
        r2.update(y_pred, y_true)

    def log_metrics(self, phase: Phase):
        r2 = self.metrics_for_phase(phase)
        self.log(f"{phase}_r2", r2.compute())
        r2.reset()

    def metrics_for_phase(self, phase: Phase):
        if phase == "train":
            return self.r2_train
        elif phase == "val":
            return self.r2_val
        else:
            return self.r2_test

    def configure_optimizers(self):
        return optim.SGD(self.output_head.parameters(), lr=self.hparams.lr)


class EncoderWarpingTask(EncoderLinearTask):
    def __init__(
        self,
        *args,
        low_dim: int,
        layer_groups: list[list[str]] | None = None,
        freeze_head: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "representation_size", "output_size"]
        )

        self.model = FastfoodWrapper(
            model=self.model, low_dim=low_dim, layer_groups=layer_groups
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_head(x)
        return x

    def configure_optimizers(self):
        if self.hparams.freeze_head:
            params = self.model.parameters()
        else:
            params = self.parameters()
        return optim.SGD(params, lr=self.hparams.lr)
