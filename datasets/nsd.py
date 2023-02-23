import os

import pandas as pd
import torch
import xarray as xr
from torchdata.datapipes.map import MapDataPipe
from torchvision import transforms
from .base import BaseDataModule, Split, Stage, ImageTransform

from datasets.utils import ImageLoaderDataPipe


class NSDDataModule(BaseDataModule):
    """
    Creates a datamodule for the NSD dataset.

    The Natural Scenes Dataset (NSD) is a collection of
    natural scene images and their fMRI responses.

    https://www.nature.com/articles/s41593-021-00962-x
    """

    def __init__(
        self,
        nsd_dir: str,
        stimuli_dir: str,
        model: str,
        train_transform: ImageTransform | None = None,
        eval_transform: ImageTransform | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        Initializes the NSD data module.

        Args:
            nsd_dir (str): Path to a directory containing the NSD voxel data (must contain a voxels.nc and metadata.csv file).
            stimuli_dir (str): Path to a directory containing the image stimuli.
            train_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for training. Defaults to None, in which case basic resizing/scaling is applied.
            eval_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for evaluation. Defaults to None, in which case basic resizing/scaling is applied.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of parallel processes loading data. Defaults to 4.
            model(str): name of model being used
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.nsd_dir = nsd_dir
        self.stimuli_dir = stimuli_dir
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.model = model

    def setup(self, stage: Stage) -> None:
        if stage in (None, "fit"):
            self._train_dataset, _ = get_nsd_dataset(
                self.nsd_dir,
                self.stimuli_dir,
                self.model,
                split="train",
                image_transform=self.train_transform,
            )
            self._val_dataset, self._n_outputs = get_nsd_dataset(
                self.nsd_dir,
                self.stimuli_dir,
                self.model,
                split="val",
                image_transform=self.eval_transform,
            )
        if stage in (None, "test"):
            self._test_dataset, self._n_outputs = get_nsd_dataset(
                self.nsd_dir,
                self.stimuli_dir,
                self.model,
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


def get_nsd_dataset(
    nsd_dir: str,
    stimuli_dir: str,
    model: str,
    split: Split,
    image_transform: ImageTransform | None = None,
) -> tuple[MapDataPipe, int]:
    """
    Returns a PyTorch datapipe for the NSD dataset with its associated image stimuli.

    Args:
        nsd_dir (str): Path to a directory containing the NSD voxel data (must contain a voxels.nc and metadata.csv file).
        stimuli_dir (str): Path to a directory containing the image stimuli.
        split (Split): One of "train", "val", or "test".
        image_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli. Defaults to None, in which case basic resizing/scaling is applied.

    Returns:
        tuple[MapDataPipe, int]: A tuple of (1) a PyTorch datapipe that returns pairs of the image stimuli as tensors and their corresponding fMRI voxel responses and (2) the number of voxels.
    """
    if image_transform is None:
        if model == "engineered_model":
            print("I found the engineered model!")
            size = 96
            image_transform = transforms.Compose(
                [
                    transforms.Resize((size,size)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)
                    ]
            )

        else:
            print("I did not find the engineered model!")
            image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    nsd_datapipe = NSDDataPipe(nsd_dir, split, model)
    image_datapipe = ImageLoaderDataPipe(stimuli_dir, nsd_datapipe.image_names)
    image_datapipe = image_datapipe.map(image_transform)
    nsd_dataset = image_datapipe.zip(nsd_datapipe)

    if split == "train":
        nsd_dataset = nsd_dataset.shuffle()

    return nsd_dataset, nsd_datapipe.n_voxels


class NSDDataPipe(MapDataPipe):
    """
    A PyTorch datapipe for the NSD dataset.
    """

    def __init__(self, data_dir: str, split: Split, model: str):
        """
        Initializes the datapipe.

        Args:
            data_dir (str): Path to a directory containing the NSD data (must contain a voxels.nc and metadata.csv file).
            split (Split): One of "train", "val", or "test".
        """
        super().__init__()

        self.data_dir = data_dir
        self.split = split

        voxels = xr.load_dataarray(os.path.join(data_dir, "voxels.nc"))
        voxels = voxels.set_index(presentation="stimulus_id")
        voxels = voxels.rename(presentation="stimulus_id")
        voxels = voxels.drop(
            [coord for coord in voxels.coords if coord != "stimulus_id"]
        )

        metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        metadata = metadata[["stimulus_id", "filename", "shared1000"]]
        metadata = metadata.set_index("stimulus_id")
        metadata = metadata.loc[voxels.stimulus_id.values]

        voxels["image_name"] = ("stimulus_id", metadata.filename)
        voxels["shared"] = ("stimulus_id", metadata.shared1000)

        if split == "train":
            if model == "engineered_model":
                print("using shared images since found enginered model")
                voxels = voxels.sel(stimulus_id=voxels.shared)
            else:
                voxels = voxels.sel(stimulus_id=~voxels.shared)
            voxels = voxels.isel(
                stimulus_id=slice(None, int(voxels.sizes["stimulus_id"] * 0.8))
            )
        elif split == "val":
            if model == "engineered_model":
                print("using shared images since found enginered model")
                voxels = voxels.sel(stimulus_id=voxels.shared)
            else:
                voxels = voxels.sel(stimulus_id=~voxels.shared)
            voxels = voxels.isel(
                stimulus_id=slice(int(voxels.sizes["stimulus_id"] * 0.8), None)
            )
        elif split == "test":
            voxels = voxels.sel(stimulus_id=voxels.shared)
        elif split == "all":
            if model == "engineered_model":
                print("using shared images since found enginered model")
                voxels = voxels.sel(stimulus_id=voxels.shared)
        voxels = voxels.drop("shared")

        self.voxels = voxels
        self.n_voxels = voxels.sizes["neuroid"]

    def __len__(self) -> int:
        return len(self.voxels)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Gets the voxel response for the given index.

        Args:
            index (int): Index of a stimulus presentation.

        Returns:
            torch.Tensor: Vector of voxel responses.
        """
        voxels = self.voxels.isel(stimulus_id=index)
        voxels = torch.from_numpy(voxels.values)
        return voxels

    @property
    def image_names(self) -> list[str]:
        """
        List of image file names (same index order as voxels).

        Returns:
            list[str]: List of image file names.
        """
        return self.voxels.image_name.values.tolist()
