import os

from PIL import Image
from torchdata.datapipes.map import MapDataPipe


class ImageLoaderDataPipe(MapDataPipe):
    """
    Generic datapipe that can load images from a directory and
    return them as PIL images.
    """

    def __init__(self, data_dir: str, image_names: list[str], **pil_open_kwargs: dict):
        """
        Initializes the loader based on a data directory and a list of image file names.

        Args:
            data_dir (str): Path to the directory containing the images.
            image_names (list[str]): List of image file names within the directory.
            **pil_open_kwargs (dict): Keyword arguments to pass to PIL's Image.open function.
        """
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
