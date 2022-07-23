import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tasks import EncoderLinearTask, EncoderWarpingTask
from torch import nn
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper


class DummyDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.n, self.d_in, self.d_out = 100, 10, 5
        x = torch.randn(self.n, self.d_in)
        w = torch.randn(self.d_in, self.d_out)
        y = x @ w + torch.randn(self.n, self.d_out) * 0.1
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


def test_encoder_linear_task():
    data = DummyDataModule()
    model = DummyModel(data.d_in)
    task = EncoderLinearTask(
        model=model,
        representation_size=model.representation_size,
        output_size=data.d_out,
    )

    trainer = pl.Trainer(
        default_root_dir="tests/saved_runs/encoder_linear",
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(monitor="val_r2", mode="max"),
            EarlyStopping(monitor="val_r2", mode="max"),
        ],
    )

    trainer.fit(task, datamodule=data)
    results = trainer.test(datamodule=data, ckpt_path="best")
