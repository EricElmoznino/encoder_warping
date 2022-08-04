from torch import nn
from abc import ABC, abstractmethod


class BaseModelLayer(nn.Module, ABC):
    """
    An abstract base class describing a model
    and the layers that can be retrieved from it.
    """
    
    @classmethod
    @property
    @abstractmethod
    def permissible_layers(cls) -> list[str]:
        """
        The names of layers that can be retrieved from this model.
        """
        raise NotImplementedError
    
    @classmethod
    @property
    @abstractmethod
    def layer_sizes(cls) -> dict[str, int]:
        """
        The size of each layer (must have one entry for each layer in `permissible_layers`).
        """
        raise NotImplementedError
    
    def __init__(self, layer: str) -> None:
        """
        Initializes the model with the specified layer.
        Stores the representation size of the model in `self.representation_size`.

        Args:
            layer (str): Name of the layer to retrieve (must be in `permissible_layers`).
        """
        super().__init__()
        assert layer in self.permissible_layers
        self.layer = layer
        self.representation_size = self.layer_sizes[layer]
    