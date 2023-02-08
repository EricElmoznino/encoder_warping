from abc import ABC, abstractmethod
from typing import Callable, Literal

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

Split = Literal["train", "val", "test", "all"]
Stage = Literal["fit", "test"] | None
ImageTransform = Callable[[Image.Image], torch.Tensor]


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    An abstract base class describing a datamodule.
    
    The reason we use this is because our PyTorch LightningModule's
    need to know about the number of outputs for a dataset,
    so this just forces us to implement that.
    """
    
    def __init__(self, batch_size: int = 64, num_workers: int = 4) -> None:
        """
        Store hyperparameters that all datamodules will need.

        Args:
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of parallel processes loading data. Defaults to 4.
        """
        print("hi the batch size is this: ", batch_size)
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.num_workers = num_workers
        
    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """
        The number of targets in the dataset (e.g. the number of fMRI voxels).
        """
        raise NotImplementedError
        
    @property
    @abstractmethod
    def train_dataset(self) -> MapDataPipe:
        """
        A PyTorch datapipe representing the training set.
        """
        raise NotImplementedError
        
    @property
    @abstractmethod
    def val_dataset(self) -> MapDataPipe:
        """
        A PyTorch datapipe representing the validation set.
        """
        raise NotImplementedError
        
    @property
    @abstractmethod
    def test_dataset(self) -> MapDataPipe:
        """
        A PyTorch datapipe representing the test set.
        """
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """
        Wraps the train_dataset in a DataLoader.

        Returns:
            DataLoader: PyTorch dataloader.
        """
        return DataLoader(
            self._train_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Wraps the val_dataset in a DataLoader.

        Returns:
            DataLoader: PyTorch dataloader.
        """
        return DataLoader(
            self._val_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Wraps the test_dataset in a DataLoader.

        Returns:
            DataLoader: PyTorch dataloader.
        """
        return DataLoader(
            self._test_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers
        )
    