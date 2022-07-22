import os

from PIL import Image
from torchdata.datapipes.map import MapDataPipe


class ImageLoaderDataPipe(MapDataPipe):
    def __init__(self, data_dir: str, image_names: list[str], **pil_open_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.image_names = image_names
        self.pil_open_kwargs = pil_open_kwargs

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> Image.Image:
        image_name = self.image_names[index]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path, **self.pil_open_kwargs)
        return image
