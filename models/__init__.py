from .torchvision_models import get_resnet18_torchvision, get_engineered_model
from .base import BaseModelLayer
from .typing import LayerGroups, ImageTransform


def get_model(
    arch: str, dataset: str, task: str, layer: str, layerwise: bool
) -> tuple[BaseModelLayer, LayerGroups, ImageTransform]:
    """
    Convenience function to grab the correct model from a configuration.

    Args:
        arch (str): Name of the model architecture.
        dataset (str): Name of the dataset the model was trained on.
        task (str): Name of the task the model was trained on.
        layer (str): Name of the layer to return from the model.

    Raises:
        ValueError: Unimplemented model specified by the configuration.

    Returns:
        tuple[BaseModelLayer, LayerGroups, ImageTransform]: A tuple containing the model,
        layer groups for non-uniform scaling in the Fastfood transform, and image transform
        to be applied to the input images.
    """
    if arch == "resnet18" and dataset == "imagenet" and task == "object-classification":
        return get_resnet18_torchvision(layer, True, layerwise)
    if arch == "resnet18" and dataset == None and task == "object-classification":
        return get_resnet18_torchvision(layer, False, layerwise)
    if arch == "engineered_model" and dataset == None and task == "object-classification":
        return get_engineered_model(layer, False, layerwise)
    else:
        raise ValueError(f"Unknown model: arch={arch}, dataset={dataset}, task={task}")
