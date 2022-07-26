import os
from typing import Callable, Literal

import pandas as pd
import torch
import xarray as xr
from PIL import Image
from torchdata.datapipes.map import MapDataPipe
from torchvision import transforms
from .base import BaseDataModule

from datasets.utils import ImageLoaderDataPipe

Split = Literal["train", "val", "test"]
Stage = Literal["fit", "test"] | None
ImageTransform = Callable[[Image.Image], torch.Tensor]


class NSDDataModule(BaseDataModule):
    def __init__(
        self,
        nsd_dir: str,
        stimuli_dir: str,
        train_transform: ImageTransform | None = None,
        eval_transform: ImageTransform | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.nsd_dir = nsd_dir
        self.stimuli_dir = stimuli_dir
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def setup(self, stage: Stage) -> None:
        if stage in (None, "fit"):
            self._train_dataset, _ = get_nsd_dataset(
                self.nsd_dir,
                self.stimuli_dir,
                split="train",
                image_transform=self.train_transform,
            )
            self._val_dataset, self._n_outputs = get_nsd_dataset(
                self.nsd_dir,
                self.stimuli_dir,
                split="val",
                image_transform=self.eval_transform,
            )
        if stage in (None, "test"):
            self._test_dataset, self._n_outputs = get_nsd_dataset(
                self.nsd_dir,
                self.stimuli_dir,
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
    split: Split,
    image_transform: ImageTransform | None = None,
) -> tuple[MapDataPipe, int]:
    if image_transform is None:
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    nsd_datapipe = NSDDataPipe(nsd_dir, split)
    image_datapipe = ImageLoaderDataPipe(stimuli_dir, nsd_datapipe.image_names)
    image_datapipe = image_datapipe.map(image_transform)
    nsd_dataset = image_datapipe.zip(nsd_datapipe)

    if split == "train":
        nsd_dataset = nsd_dataset.shuffle()

    return nsd_dataset, nsd_datapipe.n_voxels


class NSDDataPipe(MapDataPipe):
    def __init__(self, data_dir: str, split: Split):
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
            voxels = voxels.sel(stimulus_id=~voxels.shared)
            voxels = voxels.isel(
                stimulus_id=slice(None, int(voxels.sizes["stimulus_id"] * 0.8))
            )
        elif split == "val":
            voxels = voxels.sel(stimulus_id=~voxels.shared)
            voxels = voxels.isel(
                stimulus_id=slice(int(voxels.sizes["stimulus_id"] * 0.8), None)
            )
        else:
            voxels = voxels.sel(stimulus_id=voxels.shared)
        voxels = voxels.drop("shared")

        self.voxels = voxels
        self.n_voxels = voxels.sizes["neuroid"]

    def __len__(self) -> int:
        return len(self.voxels)

    def __getitem__(self, index) -> torch.Tensor:
        voxels = self.voxels.isel(stimulus_id=index)
        voxels = torch.from_numpy(voxels.values)
        return voxels

    @property
    def image_names(self):
        return self.voxels.image_name.values.tolist()
