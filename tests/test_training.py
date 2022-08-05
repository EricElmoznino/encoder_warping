import os
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tasks import (
    EncoderLinearTask,
    EncoderWarpingTask,
    ClassifierLinearTask,
    ClassifierWarpingTask,
)
from torch import nn
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper

torch.manual_seed(27)


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, classification=False) -> None:
        super().__init__()
        self.n, self.d_in, self.d_out = 100, 10, 5
        x = torch.randn(self.n, self.d_in)
        w = torch.randn(self.d_in, self.d_out)
        y = x @ w + torch.randn(self.n, self.d_out) * 0.1
        if classification:
            y = y.argmax(dim=1)
        self.data = list(zip(x, y))

    def setup(self, stage) -> None:
        train_size, val_size = int(0.6 * self.n), int(0.2 * self.n)
        self.train_dataset = SequenceWrapper(self.data[:train_size]).shuffle()
        self.val_dataset = SequenceWrapper(
            self.data[train_size : train_size + val_size]
        )
        self.test_dataset = SequenceWrapper(self.data[train_size + val_size :])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=8)


class DummyModel(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.representation_size = 20
        self.mlp = nn.Sequential(
            nn.Linear(self.d_in, self.representation_size),
            nn.Linear(self.representation_size, self.representation_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def train(
    run_name: str,
    task: pl.LightningModule,
    data: DummyDataModule,
    classification=False,
) -> float:
    metric = "accuracy" if classification else "r2"

    save_dir = os.path.join("tests/saved_runs", run_name)
    shutil.rmtree(save_dir, ignore_errors=True)

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(monitor=f"val_{metric}", mode="max"),
            EarlyStopping(monitor=f"val_{metric}", mode="max"),
        ],
    )

    trainer.fit(task, datamodule=data)
    results = trainer.test(datamodule=data, ckpt_path="best")
    return results[0][f"test_{metric}"]


class TestEncoderLinearTask:
    def test_encoder_linear(self):
        data = DummyDataModule()
        model = DummyModel(data.d_in)
        task = EncoderLinearTask(
            model=model,
            representation_size=model.representation_size,
            output_size=data.d_out,
            lr=0.1,
        )

        test_r2 = train(run_name="encoder_linear", task=task, data=data)
        assert test_r2 > 0.3


class TestEncoderWarpingTask:
    def test_encoder_warping(self):
        data = DummyDataModule()
        model = DummyModel(data.d_in)
        task = EncoderWarpingTask(
            model=model,
            low_dim=data.d_out,
            representation_size=model.representation_size,
            output_size=data.d_out,
            lr=0.1,
        )

        test_r2 = train(run_name="encoder_warping", task=task, data=data)
        assert test_r2 > 0.3

    def test_encoder_warping_layer_groups(self):
        data = DummyDataModule()
        model = DummyModel(data.d_in)
        task = EncoderWarpingTask(
            model=model,
            low_dim=data.d_out,
            layer_groups=[["mlp.0"], ["mlp.1"]],
            representation_size=model.representation_size,
            output_size=data.d_out,
            lr=0.1,
        )

        test_r2 = train(
            run_name="encoder_warping_layer_groups",
            task=task,
            data=data,
        )
        assert test_r2 > 0.3

    def test_encoder_warping_freeze_head(self):
        data = DummyDataModule()
        model = DummyModel(data.d_in)
        task = EncoderWarpingTask(
            model=model,
            low_dim=model.representation_size,
            freeze_head=True,
            representation_size=model.representation_size,
            output_size=data.d_out,
            lr=1e-2,
        )

        test_r2 = train(
            run_name="encoder_warping_freeze_head",
            task=task,
            data=data,
        )
        assert test_r2 > 0.01  # Very bad model, but still better than random


class TestClassifierLinearTask:
    def test_classifier_linear(self):
        data = DummyDataModule(classification=True)
        model = DummyModel(data.d_in)
        task = ClassifierLinearTask(
            model=model,
            representation_size=model.representation_size,
            n_classes=data.d_out,
            lr=0.1,
        )

        test_acc = train(
            run_name="classifier_linear", task=task, data=data, classification=True
        )
        assert test_acc > 0.4


class TestClassifierWarpingTask:
    def test_classifier_warping(self):
        data = DummyDataModule(classification=True)
        model = DummyModel(data.d_in)
        task = ClassifierWarpingTask(
            model=model,
            low_dim=data.d_out,
            representation_size=model.representation_size,
            n_classes=data.d_out,
            lr=0.1,
        )

        test_acc = train(
            run_name="classifier_warping", task=task, data=data, classification=True
        )
        assert test_acc > 0.4

    def test_encoder_warping_layer_groups(self):
        data = DummyDataModule(classification=True)
        model = DummyModel(data.d_in)
        task = ClassifierWarpingTask(
            model=model,
            low_dim=data.d_out,
            layer_groups=[["mlp.0"], ["mlp.1"]],
            representation_size=model.representation_size,
            n_classes=data.d_out,
            lr=0.1,
        )

        test_acc = train(
            run_name="classifier_warping_layer_groups",
            task=task,
            data=data,
            classification=True,
        )
        assert test_acc > 0.4
