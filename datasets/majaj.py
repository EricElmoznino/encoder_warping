import os

import pandas as pd
import torch
import xarray as xr
from torchdata.datapipes.map import MapDataPipe
from torchvision import transforms
from .base import BaseDataModule, Split, Stage, ImageTransform

from datasets.utils import ImageLoaderDataPipe


class MajajDataModule(BaseDataModule):
    """
    Creates a datamodule for the Majaj et al. 2015 dataset.

    The Majaj et al. 2015 dataset is a collection of
    object images and their multiunit neural recordings
    in two rhesus macaques.

    https://www.jneurosci.org/content/35/39/13402
    """

    def __init__(
        self,
        data_path: str,
        stimuli_dir: str,
        roi: str,
        train_transform: ImageTransform | None = None,
        eval_transform: ImageTransform | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        Initializes the Majaj data module.

        Args:
            data_path (str):  Path to a directory containing the Majaj neural data (*.nc file).
            stimuli_dir (str): Path to a directory containing the image stimuli.
            roi (str): Brain region of interest to use (either "IT" or "V4").
            train_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for training. Defaults to None, in which case basic resizing/scaling is applied.
            eval_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for evaluation. Defaults to None, in which case basic resizing/scaling is applied.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of parallel processes loading data. Defaults to 4.
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.data_path = data_path
        self.stimuli_dir = stimuli_dir
        self.roi = roi
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def setup(self, stage: Stage) -> None:
        if stage in (None, "fit"):
            self._train_dataset, _ = get_majaj_dataset(
                self.data_path,
                self.stimuli_dir,
                self.roi,
                split="train",
                image_transform=self.train_transform,
            )
            self._val_dataset, self._n_outputs = get_majaj_dataset(
                self.data_path,
                self.stimuli_dir,
                self.roi,
                split="val",
                image_transform=self.eval_transform,
            )
        if stage in (None, "test"):
            self._test_dataset, self._n_outputs = get_majaj_dataset(
                self.data_path,
                self.stimuli_dir,
                self.roi,
                split="test",
                image_transform=self.eval_transform,
            )

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset


def get_majaj_dataset(
    data_path: str,
    stimuli_dir: str,
    roi: str,
    split: Split,
    image_transform: ImageTransform | None = None,
) -> tuple[MapDataPipe, int]:
    """
    Returns a PyTorch datapipe for the Majaj dataset with its associated image stimuli.

    Args:
        data_path (str): Path to a directory containing the Majaj neural data (*.nc file).
        stimuli_dir (str): Path to a directory containing the image stimuli.
        roi (str): Brain region of interest to use (either "IT" or "V4").
        split (Split): One of "train", "val", or "test".
        image_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli. Defaults to None, in which case basic resizing/scaling is applied.

    Returns:
        tuple[MapDataPipe, int]: A tuple of (1) a PyTorch datapipe that returns pairs of the image stimuli as tensors and their corresponding neural responses and (2) the number of recording probes.
    """
    if image_transform is None:
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    majaj_datapipe = MajajDataPipe(data_path, roi, split)
    image_datapipe = ImageLoaderDataPipe(stimuli_dir, majaj_datapipe.image_names)
    image_datapipe = image_datapipe.map(image_transform)
    majaj_dataset = image_datapipe.zip(majaj_datapipe)

    if split == "train":
        majaj_dataset = majaj_dataset.shuffle()

    return majaj_dataset, majaj_datapipe.n_neurons


class MajajDataPipe(MapDataPipe):
    """
    A PyTorch datapipe for the Majaj dataset.
    """

    def __init__(self, data_path: str, roi: str, split: Split):
        """
        Initializes the datapipe.

        Args:
            data_path (str): Path to a directory containing the Majaj neural data (*.nc file).
            roi (str): Brain region of interest to use (either "IT" or "V4").
            split (Split): One of "train", "val", or "test".
        """
        super().__init__()

        self.data_path = data_path
        self.roi = roi
        self.split = split

        neurons = xr.load_dataarray(data_path)
        neurons = neurons.sel(neuroid=neurons.region == roi)

        if split == "train":
            neurons = neurons.isel(
                image_id=slice(None, int(neurons.sizes["image_id"] * 0.6))
            )
        elif split == "val":
            neurons = neurons.isel(
                image_id=slice(
                    int(neurons.sizes["image_id"] * 0.6),
                    int(neurons.sizes["image_id"] * 0.8),
                )
            )
        else:
            neurons = neurons.isel(
                image_id=slice(int(neurons.sizes["image_id"] * 0.8), None)
            )

        self.neurons = neurons
        self.n_neurons = neurons.sizes["neuroid"]

    def __len__(self) -> int:
        return len(self.neurons)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Gets the neural response for the given index.

        Args:
            index (int): Index of a stimulus presentation.

        Returns:
            torch.Tensor: Vector of neural responses.
        """
        neurons = self.neurons.isel(image_id=index)
        neurons = torch.from_numpy(neurons.values)
        return neurons

    @property
    def image_names(self):
        """
        List of image file names (same index order as neural recordings).

        Returns:
            list[str]: List of image file names.
        """
        return self.neurons.image_name.values.tolist()
