import os
import random

import torch
from torchdata.datapipes.map import MapDataPipe
from torchvision import transforms
from .base import BaseDataModule, Split, Stage, ImageTransform

from datasets.utils import ImageLoaderDataPipe


class ImageFolderDataModule(BaseDataModule):
    """
    Creates a datamodule from a folder of images that has
    subfolders for different classes.
    """

    def __init__(
        self,
        data_dir: str,
        limit_classes: int | None = None,
        limit_samples: int | None = None,
        train_transform: ImageTransform | None = None,
        eval_transform: ImageTransform | None = None,
        random_seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        Initializes the ImageFolder data module.

        Args:
            data_dir (str): Path to a directory containing the image dataset.
            limit_classes (int | None, optional): Limit the number of classes to use. Defaults to None, in which case all classes are used.
            limit_samples (int | None, optional): Limit the number of samples to use per class. Defaults to None, in which case all class samples are used.
            train_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for training. Defaults to None, in which case basic resizing/scaling is applied.
            eval_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for evaluation. Defaults to None, in which case basic resizing/scaling is applied.
            random_seed (int): Random seed for consistency in sampled classes/samples across calls. Defaults to 0.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of parallel processes loading data. Defaults to 4.
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.data_dir = data_dir
        self.limit_classes = limit_classes
        self.limit_samples = limit_samples
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.random_seed = random_seed

    def setup(self, stage: Stage) -> None:
        if stage in (None, "fit"):
            self._train_dataset, _ = get_image_folder_dataset(
                self.data_dir,
                split="train",
                limit_classes=self.limit_classes,
                limit_samples=self.limit_samples,
                image_transform=self.train_transform,
                random_seed=self.random_seed,
            )
            self._val_dataset, self._n_outputs = get_image_folder_dataset(
                self.data_dir,
                split="val",
                limit_classes=self.limit_classes,
                limit_samples=self.limit_samples,
                image_transform=self.eval_transform,
                random_seed=self.random_seed,
            )
        if stage in (None, "test"):
            self._test_dataset, self._n_outputs = get_image_folder_dataset(
                self.data_dir,
                split="test",
                limit_classes=self.limit_classes,
                limit_samples=self.limit_samples,
                image_transform=self.eval_transform,
                random_seed=self.random_seed,
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


def get_image_folder_dataset(
    data_dir: str,
    split: Split,
    limit_classes: int | None = None,
    limit_samples: int | None = None,
    image_transform: ImageTransform | None = None,
    random_seed: int = 0,
) -> tuple[MapDataPipe, int]:
    """
    Creates a dataset from a folder of images that has
    subfolders for different classes.

    Args:
        data_dir (str): Path to a directory containing the image dataset.
        split (Split): Split to use.
        limit_classes (int | None, optional): Limit the number of classes to use. Defaults to None, in which case all classes are used.
        limit_samples (int | None, optional): Limit the number of samples to use per class. Defaults to None, in which case all class samples are used.
        image_transform (ImageTransform | None, optional): Image preprocessing to apply when loading the stimuli for training. Defaults to None, in which case basic resizing/scaling is applied.
        random_seed (int): Random seed for consistency in sampled classes/samples across calls. Defaults to 0.

    Returns:
        tuple[MapDataPipe, int]: A tuple of (1) a PyTorch datapipe that returns pairs of the image stimuli as tensors and their corresponding class labels and (2) the number of classes.
    """
    if image_transform is None:
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    image_folder_datapipe = ImageFolderDataPipe(
        data_dir, split, limit_classes, limit_samples, random_seed
    )
    image_datapipe = ImageLoaderDataPipe(data_dir, image_folder_datapipe.image_names)
    image_datapipe = image_datapipe.map(image_transform)
    image_dataset = image_datapipe.zip(image_folder_datapipe)

    if split == "train":
        image_dataset = image_dataset.shuffle()

    return image_dataset, image_folder_datapipe.num_classes


class ImageFolderDataPipe(MapDataPipe):
    """
    A PyTorch datapipe for an arbitrary folder of images that has
    subfolders for different classes.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        limit_classes: int | None = None,
        limit_samples: int | None = None,
        random_seed: int = 0,
    ) -> None:
        """
        Initializes the datapipe.

        Args:
            data_dir (str): Path to a directory containing the image dataset.
            split (str): One of 'train', 'valid', or 'test'.
            limit_classes (int | None, optional): Limit the number of classes to use. Defaults to None, in which case all classes are used.
            limit_samples (int | None, optional): Limit the number of samples to use per class. Defaults to None, in which case all class samples are used.
            random_seed (int): Random seed for consistency in sampled classes/samples across calls. Defaults to 0.
        """
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.limit_classes = limit_classes
        self.limit_samples = limit_samples
        self.random_seed = random_seed

        classes = os.listdir(self.data_dir)
        if self.limit_classes is not None:
            rand = random.Random(self.random_seed)
            random.shuffle(classes)
            classes = classes[: self.limit_classes]

        image_names = []
        class_labels = []
        for i, c in enumerate(classes):
            class_dir = os.path.join(self.data_dir, c)
            class_samples = os.listdir(class_dir)
            class_samples = [os.path.join(c, s) for s in class_samples]

            if self.limit_samples is not None:
                rand = random.Random(self.random_seed)
                random.shuffle(class_samples)
                class_samples = class_samples[: self.limit_samples]

            if split == "train":
                class_samples = class_samples[: int(len(class_samples) * 0.6)]
            elif split == "val":
                class_samples = class_samples[
                    int(len(class_samples) * 0.6) : int(len(class_samples) * 0.8)
                ]
            else:
                class_samples = class_samples[int(len(class_samples) * 0.8) :]

            image_names += class_samples
            class_labels += [i] * len(class_samples)

        self.image_names = image_names
        self.class_labels = class_labels
        self.num_classes = len(classes)
        self.class_name_map = {i: c for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int) -> torch.LongTensor:
        """
        Gets the image class for the given index.

        Args:
            index (int): Index of the image.

        Returns:
            torch.LongTensor: Scalar tensor containing the class number.
        """
        class_label = self.class_labels[index]
        class_label = torch.tensor(class_label)
        return class_label
