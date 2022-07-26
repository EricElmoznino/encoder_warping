import numpy as np
import torch
from low_dim import FastfoodWrapper
from models import get_model
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


class TestResnet18:

    test_image = Image.open("tests/images/frond.jpg")
    layers = ["maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]

    def test_correct_output(self):
        reference = create_feature_extractor(
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval(), self.layers
        )
        reference_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        with torch.no_grad():
            reference_output = reference(
                reference_transform(self.test_image).unsqueeze(dim=0)
            )
        reference_output["avgpool"] = reference_output["avgpool"].flatten(1)

        for layer in self.layers:
            model, _, transform = get_model("resnet18", layer)
            with torch.no_grad():
                output = model(transform(self.test_image).unsqueeze(dim=0))

            assert (output == reference_output[layer]).all()
            assert np.prod(output.shape) == model.output_size

    def test_fastfood_output(self):
        model, layer_groups, transform = get_model("resnet18", "avgpool")
        ff_model = FastfoodWrapper(model, low_dim=50, layer_groups=layer_groups)
        with torch.no_grad():
            output = model(transform(self.test_image).unsqueeze(dim=0))
            ff_output = ff_model(transform(self.test_image).unsqueeze(dim=0))

        assert (output == ff_output).all()

    def test_fastfood_grad(self):
        model, layer_groups, transform = get_model("resnet18", "avgpool")
        model = FastfoodWrapper(model, low_dim=50, layer_groups=layer_groups)
        output = model(transform(self.test_image).unsqueeze(dim=0))
        output.sum().backward()
        assert (
            model.low_dim_params.grad is not None
            and model.group_scalers.grad is not None
        )
