from abc import ABC, abstractmethod
from typing import Callable, Literal

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

Split = Literal["train", "val", "test"]
Stage = Literal["fit", "test"] | None
ImageTransform = Callable[[Image.Image], torch.Tensor]


class BaseDataModule(pl.LightningDataModule, ABC):
    
    def __init__(self, batch_size: int = 64, num_workers: int = 4) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.num_workers = num_workers
        
    @property
    @abstractmethod
    def n_outputs(self) -> int:
        raise NotImplementedError
        
    @property
    @abstractmethod
    def train_dataset(self) -> MapDataPipe:
        raise NotImplementedError
        
    @property
    @abstractmethod
    def val_dataset(self) -> MapDataPipe:
        raise NotImplementedError
        
    @property
    @abstractmethod
    def test_dataset(self) -> MapDataPipe:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers
        )
    