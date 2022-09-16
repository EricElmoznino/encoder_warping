import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from .typing import LayerGroups, ImageTransform
from .base import BaseModelLayer


def get_resnet18_torchvision(
    layer: str,
) -> tuple[BaseModelLayer, LayerGroups, ImageTransform]:
    """
    Get the specified layer from a ResNet18 model trained on ImageNet classification.

    Args:
        layer (str): Name of the layers to retrieve from the model.

    Returns:
        tuple[BaseModelLayer, LayerGroups, ImageTransform]: A tuple containing the model,
        layer groups for non-uniform scaling in the Fastfood transform, and image transform
        to be applied to the input images.
    """
    assert layer in ResNet18Layer.permissible_layers
    weights = ResNet18_Weights.IMAGENET1K_V1

    model = ResNet18Layer(layer, weights)

    # Only need layers with parameters in these groups
    layer_groups = [["conv1", "bn1"], "layer1", "layer2", "layer3", "layer4"]
    layer_groups = layer_groups[: ResNet18Layer.permissible_layers.index(layer) + 1]

    image_transform = weights.transforms()

    return model, layer_groups, image_transform


class ResNet18Layer(BaseModelLayer):

    permissible_layers = [
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
    ]

    layer_sizes = {
        "maxpool": 200704,
        "layer1": 200704,
        "layer2": 100352,
        "layer3": 50176,
        "layer4": 25088,
        "avgpool": 512,
    }

    def __init__(self, layer: str, weights=None) -> None:
        super().__init__(layer)

        base = resnet18(weights=weights)
        layer_idx = self.permissible_layers.index(layer)
        self.conv1, self.bn1, self.relu, self.maxpool = (
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        if layer_idx >= 1:
            self.layer1 = base.layer1
        if layer_idx >= 2:
            self.layer2 = base.layer2
        if layer_idx >= 3:
            self.layer3 = base.layer3
        if layer_idx >= 4:
            self.layer4 = base.layer4
        if layer_idx == 5:
            self.avgpool = base.avgpool

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.layer == "maxpool":
            return x
        x = self.layer1(x)
        if self.layer == "layer1":
            return x
        x = self.layer2(x)
        if self.layer == "layer2":
            return x
        x = self.layer3(x)
        if self.layer == "layer3":
            return x
        x = self.layer4(x)
        if self.layer == "layer4":
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.layer == "avgpool":
            return x
        raise ValueError(f"Invalid layer: {self.layer}")
