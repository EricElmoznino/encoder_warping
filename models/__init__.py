from .torchvision_models import get_resnet18_torchvision
from .base import BaseModelLayer
from .typing import LayerGroups, ImageTransform


def get_model(
    arch: str, dataset: str, task: str, layer: str
) -> tuple[BaseModelLayer, LayerGroups, ImageTransform]:
    if arch == "resnet18" and dataset == "imagenet" and task == "object-classification":
        return get_resnet18_torchvision(layer)
    else:
        raise ValueError(f"Unknown model: arch={arch}, dataset={dataset}, task={task}")
