import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, convnext_base, ConvNeXt_Base_Weights, vit_b_16, ViT_B_16_Weights
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
    # layer_groups = layer_groups[: ResNet18Layer.permissible_layers.index(layer) + 1]

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
        "layer1": 200704, #64
        "layer2": 100352, #128
        "layer3": 50176, #256
        "layer4": 25088, #512
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

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
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
            x = self.avgpool(x)
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


def get_convnext_torchvision(
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
    assert layer in ConvNextLayer.permissible_layers
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1

    model = ConvNextLayer(layer, weights)

    # Only need layers with parameters in these groups
    layer_groups = ["features", "avgpool"]

    image_transform = weights.transforms()

    return model, layer_groups, image_transform

class ConvNextLayer(BaseModelLayer):

    permissible_layers = [
        "features",
        "avgpool",
    ]

    layer_sizes = {
        "features": 64, 
        "avgpool": 1024,
    }

    def __init__(self, layer: str, weights=None) -> None:
        super().__init__(layer)

        base = convnext_base(weights=weights)
        print(base)

        self.features = base.features
        self.avgpool = base.avgpool

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.layer == "features":
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.layer == "avgpool":
            print(x.shape)
            return x
        return x
        raise ValueError(f"Invalid layer: {self.layer}")

def get_vit16_torchvision(
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
    assert layer in vitLayer.permissible_layers
    weights = ViT_B_16_Weights.IMAGENET1K_V1

    model = vitLayer(layer, weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    layer_groups = [
    "class_token",
    "conv_proj",
    ["encoder.pos_embedding", "encoder.ln"],
    ["encoder.layers.encoder_layer_0", "encoder.layers.encoder_layer_1", "encoder.layers.encoder_layer_2"],
    ["encoder.layers.encoder_layer_3", "encoder.layers.encoder_layer_4", "encoder.layers.encoder_layer_5"],
    ["encoder.layers.encoder_layer_6", "encoder.layers.encoder_layer_7", "encoder.layers.encoder_layer_8"],
    ["encoder.layers.encoder_layer_9", "encoder.layers.encoder_layer_10", "encoder.layers.encoder_layer_11"],
    "adp_avgpool"
]

    image_transform = weights.transforms()

    return model, layer_groups, image_transform

class vitLayer(BaseModelLayer):

    permissible_layers = [
        "encoder",
        "adp_avgpool"
    ]

    layer_sizes = {
        "encoder": 768, 
        "adp_avgpool": 768 #1024,
    }

    def __init__(self, layer: str, weights=None) -> None:
        super().__init__(layer)

        base = vit_b_16(weights=weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = base.to(device)

        self._process_input = base._process_input
        self.class_token = base.class_token
        self.encoder = base.encoder
        self.adp_avgpool = nn.AdaptiveAvgPool1d(1024)

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        if self.layer == "encoder":
            return x

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        # x = self.adp_avgpool(x)
        if self.layer == "adp_avgpool":
            return x

        return x
        raise ValueError(f"Invalid layer: {self.layer}")
