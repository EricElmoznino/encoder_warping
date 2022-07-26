from .torchvision_models import get_resnet18_torchvision
from .base import BaseModelLayer
from .typing import LayerGroups, ImageTransform


def get_model(
    architecture: str, layer: str
) -> tuple[BaseModelLayer, LayerGroups, ImageTransform]:
    if architecture == "resnet18":
        return get_resnet18_torchvision(layer)
    else:
        raise ValueError(f"Unknown architecture {architecture}")
