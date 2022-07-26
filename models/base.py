from torch import nn
from abc import ABC, abstractmethod


class BaseModelLayer(nn.Module, ABC):
    
    @classmethod
    @property
    @abstractmethod
    def permissible_layers(cls) -> list[str]:
        raise NotImplementedError
    
    @classmethod
    @property
    @abstractmethod
    def layer_sizes(cls) -> dict[str, int]:
        raise NotImplementedError
    
    def __init__(self, layer: str, weights=None) -> None:
        super().__init__()
        assert layer in self.permissible_layers
        self.layer = layer
        self.output_size = self.layer_sizes[layer]
    