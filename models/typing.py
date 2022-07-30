from typing import Callable

import torch
from PIL import Image

LayerGroups = list[str | list[str]]
ImageTransform = Callable[[Image.Image], torch.Tensor]
