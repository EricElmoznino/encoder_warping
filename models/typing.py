from typing import Callable

import torch
from PIL import Image

LayerGroups = list[list[str]]
ImageTransform = Callable[[Image.Image], torch.Tensor]
