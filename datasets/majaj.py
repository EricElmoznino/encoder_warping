import os

import pandas as pd
import torch
import xarray as xr
from torchdata.datapipes.map import MapDataPipe
from torchvision import transforms
from .base import BaseDataModule, Split, Stage, ImageTransform

from datasets.utils import ImageLoaderDataPipe


class MajajDataModule(BaseDataModule):
    def __init__(
        self,
        neural_dir: str,
        stimuli_dir: str,
        roi: str,
        train_transform: ImageTransform | None = None,
        eval_transform: ImageTransform | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.neural_dir = neural_dir
        self.stimuli_dir = stimuli_dir
        self.roi = roi
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def setup(self, stage: Stage) -> None:
        if stage in (None, "fit"):
            self._train_dataset, _ = get_majaj_dataset(
                self.neural_dir,
                self.stimuli_dir,
                self.roi,
                split="train",
                image_transform=self.train_transform,
            )
            self._val_dataset, self._n_outputs = get_majaj_dataset(
                self.neural_dir,
                self.stimuli_dir,
                self.roi,
                split="val",
                image_transform=self.eval_transform,
            )
        if stage in (None, "test"):
            self._test_dataset, self._n_outputs = get_majaj_dataset(
                self.neural_dir,
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
    neural_dir: str,
    stimuli_dir: str,
    roi: str,
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

    majaj_datapipe = MajajDataPipe(neural_dir, roi, split)
    image_datapipe = ImageLoaderDataPipe(stimuli_dir, majaj_datapipe.image_names)
    image_datapipe = image_datapipe.map(image_transform)
    majaj_dataset = image_datapipe.zip(majaj_datapipe)

    if split == "train":
        majaj_dataset = majaj_dataset.shuffle()

    return majaj_dataset, majaj_datapipe.n_neurons


class MajajDataPipe(MapDataPipe):
    def __init__(self, data_dir: str, roi: str, split: Split):
        super().__init__()

        self.data_dir = data_dir
        self.roi = roi
        self.split = split

        neural = xr.load_dataarray(os.path.join(data_dir, "neural.nc"))
        neural = neural.sel(neuroid=neural.region == roi)
        neural = neural.drop([coord for coord in neural.coords if coord != "image_id"])

        metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        metadata = metadata[["image_id", "filename", "object_name"]]
        metadata = metadata.set_index("image_id")

        neural["image_name"] = ("image_id", metadata.filename)
        neural["object_name"] = ("image_id", metadata.object_name)

        if split == "train":
            neural = neural.isel(
                image_id=slice(None, int(neural.sizes["image_id"] * 0.6))
            )
        elif split == "val":
            neural = neural.isel(
                image_id=slice(
                    int(neural.sizes["image_id"] * 0.6),
                    int(neural.sizes["image_id"] * 0.8),
                )
            )
        else:
            neural = neural.isel(
                image_id=slice(int(neural.sizes["image_id"] * 0.8), None)
            )

        self.neural = neural
        self.n_neurons = neural.sizes["neuroid"]

    def __len__(self) -> int:
        return len(self.neural)

    def __getitem__(self, index) -> torch.Tensor:
        neurons = self.neural.isel(image_id=index)
        neurons = torch.from_numpy(neurons.values)
        return neurons

    @property
    def image_names(self):
        return self.neural.image_name.values.tolist()
