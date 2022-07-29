import torch
from torchvision import transforms
from datasets.nsd import NSDDataPipe, NSDDataModule
from datasets.majaj import MajajDataPipe, MajajDataModule

# Replace these paths for your own tests
nsd_dir = "/home/eric/datasets/natural_scenes_dataset"
nsd_stimuli_dir = "/home/eric/datasets/natural_scenes_dataset/stimuli"
majaj_dir = "/home/eric/datasets/majaj"
majaj_stimuli_dir = "/home/eric/datasets/majaj/stimuli"


class TestNSDDataModule:
    def test_setup(self):
        dm = NSDDataModule(nsd_dir, nsd_stimuli_dir)
        dm.setup(stage=None)

        assert (
            len(dm.train_dataset) > 0
            and len(dm.val_dataset) > 0
            and len(dm.test_dataset) > 0
        )
        assert dm.n_outputs > 1

    def test_shapes_and_types(self):
        dm = NSDDataModule(nsd_dir, nsd_stimuli_dir)
        dm.setup(stage="test")
        dl = dm.test_dataloader()
        image, voxels = next(iter(dl))
        batch_size = dm.hparams.batch_size
        assert image.shape == (batch_size, 3, 224, 224)
        assert image.dtype == torch.float32
        assert voxels.shape[0] == batch_size and voxels.shape[1] > 1
        assert voxels.dtype == torch.float32

    def test_image_transform(self):
        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
        dm = NSDDataModule(nsd_dir, nsd_stimuli_dir, eval_transform=transform)
        dm.setup(stage="test")
        image, _ = next(iter(dm.test_dataloader()))
        assert image.shape == (dm.hparams.batch_size, 3, 64, 64)


class TestNSDDataPipe:
    def test_not_empty(self):
        for split in ["train", "val", "test"]:
            dp = NSDDataPipe(nsd_dir, split)
            assert len(dp) > 0

    def test_image_names(self):
        dp = NSDDataPipe(nsd_dir, "test")
        image_names = dp.image_names
        assert isinstance(image_names, list)
        assert isinstance(image_names[0], str)
        assert len(image_names) == len(dp)

    def test_get_item(self):
        dp = NSDDataPipe(nsd_dir, "test")
        voxels = dp[0]
        assert voxels.ndim == 1
        assert voxels.dtype == torch.float32

    def test_no_overlap(self):
        train = NSDDataPipe(nsd_dir, split="train")
        val = NSDDataPipe(nsd_dir, split="val")
        test = NSDDataPipe(nsd_dir, split="test")
        all_images = train.image_names + val.image_names + test.image_names
        assert len(all_images) == len(set(all_images))


class TestMajajDataModule:
    def test_setup(self):
        dm = MajajDataModule(majaj_dir, majaj_stimuli_dir, "IT")
        dm.setup(stage=None)

        assert (
            len(dm.train_dataset) > 0
            and len(dm.val_dataset) > 0
            and len(dm.test_dataset) > 0
        )
        assert dm.n_outputs > 1

    def test_shapes_and_types(self):
        dm = MajajDataModule(majaj_dir, majaj_stimuli_dir, "IT")
        dm.setup(stage="test")
        dl = dm.test_dataloader()
        image, neurons = next(iter(dl))
        batch_size = dm.hparams.batch_size
        assert image.shape == (batch_size, 3, 224, 224)
        assert image.dtype == torch.float32
        assert neurons.shape[0] == batch_size and neurons.shape[1] > 1
        assert neurons.dtype == torch.float32

    def test_image_transform(self):
        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
        dm = MajajDataModule(
            majaj_dir, majaj_stimuli_dir, "IT", eval_transform=transform
        )
        dm.setup(stage="test")
        image, _ = next(iter(dm.test_dataloader()))
        assert image.shape == (dm.hparams.batch_size, 3, 64, 64)


class TestMajajDataPipe:
    def test_not_empty(self):
        for split in ["train", "val", "test"]:
            dp_it = MajajDataPipe(majaj_dir, "IT", split)
            dp_v4 = MajajDataPipe(majaj_dir, "V4", split)
            assert len(dp_it) > 0 and len(dp_v4) > 0

    def test_image_names(self):
        dp = MajajDataPipe(majaj_dir, "IT", "test")
        image_names = dp.image_names
        assert isinstance(image_names, list)
        assert isinstance(image_names[0], str)
        assert len(image_names) == len(dp)

    def test_get_item(self):
        dp = MajajDataPipe(majaj_dir, "IT", "test")
        neurons = dp[0]
        assert neurons.ndim == 1
        assert neurons.dtype == torch.float32

    def test_no_overlap(self):
        train = MajajDataPipe(majaj_dir, "IT", "train")
        val = MajajDataPipe(majaj_dir, "IT", "val")
        test = MajajDataPipe(majaj_dir, "IT", "test")
        all_images = train.image_names + val.image_names + test.image_names
        assert len(all_images) == len(set(all_images))
