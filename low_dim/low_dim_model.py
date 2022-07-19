from typing import Any, Iterable, Callable
import re
from math import sqrt
from more_itertools import collapse
import torch
from torch import nn
from low_dim.delta_module import DeltaModule, DeltaLinear, DeltaConv2d


TrainableLayer = list[str] | list[list[str]]

layer_delta_mappings = {nn.Linear: DeltaLinear, nn.Conv2d: DeltaConv2d}


class LowDimModel(nn.Module):
    def __init__(self, model: nn.Module, dim: int, trainable_layers: TrainableLayer):
        super().__init__()

        # Initialize scalers for each group of layers (if layers are grouped)
        assert len(trainable_layers) > 0
        layers_are_grouped = isinstance(trainable_layers[0], list)
        if layers_are_grouped:
            assert (
                len(trainable_layers) > 1
            ), "Only one group of layers found: use a single list of layers"
            assert all(
                isinstance(group, list) for group in trainable_layers
            ), "All groups of layers must be lists of strings"
            self.num_layer_groups = len(trainable_layers)
            self.layer_scalers = nn.Parameter(torch.ones(self.num_layer_groups))
        else:
            self.num_layer_groups = 0
            self.layer_scalers = None

        # Initialize model and low-dimensional projection of parameters
        self.dim = dim
        self.model = model.eval()
        self.layer_names = list(collapse(trainable_layers))
        self.low_dim_params = nn.Parameter(
            torch.zeros(self.dim - self.num_layer_groups)
        )

        # Replace layers with delta modules and store them for later use
        self.layers = self._replace_trainable_layers(self.layer_names)
        self.layer_num_params = {
            layer: module.trainable_num_params for layer, module in self.layers.items()
        }
        self.total_layer_params = sum(self.layer_num_params.values())
        if layers_are_grouped:
            self.layer_to_group_idx = {
                layer: group_idx
                for group_idx, layer_group in enumerate(trainable_layers)
                for layer in layer_group
            }
        else:
            self.layer_to_group_idx = None

        # Create low-to-high dimensional projection matrix
        # TODO: replace with the Fastfood transform (Le et al., 2013)
        self.low_to_high_proj = torch.normal(
            mean=0,
            std=1,
            size=(self.dim - self.num_layer_groups, self.total_layer_params),
        ) / sqrt(self.total_layer_params)

    def forward(self, *args, **kwargs) -> Any:
        if self.training:
            # Project low-dimensional parameters to high-dimensional space
            # to update the model's parameters and to get backpropagated gradients
            self.set_layer_params()
        return self.model(*args, **kwargs)

    def set_layer_params(self) -> None:
        """
        Sets the parameters of the model's delta modules to the high-dimensional
        projection of the low-dimensional parameters.
        """
        # Project low-dimensional parameters to high-dimensional space
        # TODO: replace with the Fastfood transform (Le et al., 2013)
        high_dim_deltas = self.low_dim_params @ self.low_to_high_proj

        # Split high-dimensional parameter deltas into their individual layers
        high_dim_deltas = high_dim_deltas.split(list(self.layer_num_params.values()))
        high_dim_deltas = {
            layer: params for layer, params in zip(self.layer_names, high_dim_deltas)
        }

        # Scale each layer's deltas by its group scaler (if layers are grouped)
        if self.layer_scalers is not None:
            for layer in self.layer_names:
                group_idx = self.layer_to_group_idx[layer]
                high_dim_deltas[layer] = (
                    high_dim_deltas[layer] * self.layer_scalers[group_idx]
                )

        # Set the parameters of the delta modules
        for layer in self.layer_names:
            delta_module = self.layers[layer]
            delta = high_dim_deltas[layer]
            delta_module.set_params(delta)

    def _replace_trainable_layers(
        self, trainable_layers: Iterable[str]
    ) -> dict[str, DeltaModule]:
        # Check that all trainable layers exist and are valid
        model_layers = dict(self.model.named_modules())
        for layer_name in trainable_layers:
            assert (
                layer_name in model_layers
            ), f'Layer "{layer_name}" not found in model'
            assert (
                type(model_layers[layer_name]) in layer_delta_mappings
            ), f'DeltaModule for layer "{layer_name}" not implemented'

        # Replace layers with delta modules
        delta_modules = {}
        for layer_name in trainable_layers:
            delta_module = self._replace_layer_with_delta(layer_name)
            delta_modules[layer_name] = delta_module

        return delta_modules

    def _replace_layer_with_delta(self, layer_name: str) -> DeltaModule:
        """
        Replaces a layer with its equivalent delta module. Note that this does the
        replacement in-place, but also returns the delta module so that the calling
        function can access its attributes directly rather than having to re-fetch
        it from the model.
        Args:
            layer_name (str): The name of the layer to replace with a delta module.

        Returns:
            _type_: _description_
        """
        layer_name_hierarchy = re.split(r"\.\d+\.", layer_name)
        layer_index_hierarchy = [int(x) for x in re.findall(r"\.(\d+)\.", layer_name)]

        def swap_attr(module: nn.Module, attr_str: str) -> DeltaModule:
            old_val = getattr(module, attr_str)
            old_type = type(old_val)
            new_type = layer_delta_mappings[old_type]
            new_val = new_type(old_val)
            setattr(module, attr_str, new_val)
            return new_val

        if len(layer_index_hierarchy) == 0:  # No nn.Sequential's
            return swap_attr(self.model, layer_name)
        else:
            module = self.model
            for name, index in zip(layer_name_hierarchy[:-1], layer_index_hierarchy):
                module = getattr(module, name)[index]
            return swap_attr(module, layer_name_hierarchy[-1])
