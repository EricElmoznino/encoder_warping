from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F

Params = dict[str, Tensor]
Inputs = Tensor | tuple[Tensor, ...] | dict[str, Tensor]


class DeltaModule(nn.Module, ABC):
    def __init__(self, module: nn.Module, trainable: list[str]):
        super(DeltaModule, self).__init__()

        params = module.named_parameters()
        params = {n: p.data for n, p in params}

        for name in trainable:
            assert name in params

        self.trainable_params_initial: Params = {n: params[n] for n in trainable}
        self.constant_params: Params = {
            n: params[n] for n in params if n not in trainable
        }

        self.trainable_names = trainable
        self.trainable_shapes = {
            n: p.shape for n, p in self.trainable_params_initial.items()
        }
        self.trainable_sizes = {n: s.numel() for n, s in self.trainable_shapes.items()}
        self.trainable_size_idxs = np.cumsum(list(self.trainable_sizes.values()))
        self.trainable_num_params = self.trainable_size_idxs[-1]

        self.trainable_params: Params = self.trainable_params_initial

    @abstractmethod
    def forward_func(self, inputs: Inputs, params: Params) -> Any:
        pass

    def set_params(self, delta: Tensor) -> None:
        assert delta.ndim == 1 and len(delta) == self.trainable_num_params

        params, i_minus_1 = {}, 0
        for name, i in zip(self.trainable_names, self.trainable_size_idxs):
            param = delta[i_minus_1:i]
            param = param.view(self.trainable_shapes[name])
            param = param + self.trainable_params_initial[name]
            params[name] = param
            i_minus_1 = i

        self.trainable_params = params

    def forward(self, inputs: Inputs) -> Any:
        params = {**self.trainable_params, **self.constant_params}
        return self.forward_func(inputs, params)


class DeltaLinear(DeltaModule):
    class LinearParams(TypedDict, total=False):
        weight: Tensor
        bias: Tensor

    def __init__(self, module: nn.Linear, trainable: list[str]):
        super(DeltaLinear, self).__init__(module, trainable)

        self.in_features = module.in_features
        self.out_features = module.out_features

    def forward_func(self, inputs: Tensor, params: LinearParams):
        w = params["weight"]
        b = params["bias"] if "bias" in params else None
        return F.linear(input=inputs, weight=w, bias=b)


class DeltaConv2d(DeltaModule):
    class Conv2dParams(TypedDict, total=False):
        weight: Tensor
        bias: Tensor

    def __init__(self, module: nn.Conv2d, trainable: list[str]):
        super(DeltaConv2d, self).__init__(module, trainable)

        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups

        assert module.padding_mode == "zeros"  # TODO: support this later

    def forward_func(self, inputs: Tensor, params: Conv2dParams):
        w = params["weight"]
        b = params["bias"] if "bias" in params else None
        return F.conv2d(
            input=inputs,
            weight=w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
