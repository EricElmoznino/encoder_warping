import torch
from torchvision import transforms
from datasets import NSDDataModule
from datasets.nsd import NSDDataPipe

# Replace these paths for your own tests
nsd_dir = "/home/eric/datasets/natural_scenes_dataset"
nsd_stimuli_dir = "/home/eric/datasets/natural_scenes_dataset/stimuli"


class TestNSDDataModule:
    def test_setup(self):
        dm = NSDDataModule(nsd_dir, nsd_stimuli_dir)
        dm.setup(stage=None)

        assert (
            len(dm._train_dataset) > 0
            and len(dm._val_dataset) > 0
            and len(dm._test_dataset) > 0
        )
        assert dm._n_outputs > 1

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
