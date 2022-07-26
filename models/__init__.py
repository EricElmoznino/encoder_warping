from torch import nn
from .torchvision_models import get_resnet18_torchvision
from .typing import LayerGroups, ImageTransform


def get_model(
    architecture: str, layer: str
) -> tuple[nn.Module, LayerGroups, ImageTransform]:
    if architecture == "resnet18":
        return get_resnet18_torchvision(layer)
    else:
        raise ValueError(f"Unknown architecture {architecture}")
