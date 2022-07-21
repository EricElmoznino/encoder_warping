"""
Adapted from:
    https://github.com/jgamper/intrinsic-dimensionality/blob/master/intrinsic/fastfood.py 
Implements the work in:
    https://arxiv.org/abs/1804.08838
"""

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FastfoodWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        low_dim: int,
        layer_groups: list[list[str]] | None = None,
    ):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model using FastFood transform
        :param module: pytorch nn.Module to wrap
        :param low_dim: dimensionality of the low-dimensional parameters used to optimize the model
        :param layer_groups: group layers who's parameter deltas will be scaled by learnable weights
        """
        super().__init__()

        # Check that arguments are valid
        if layer_groups is not None:
            assert (
                len(layer_groups) > 1
            ), "layer_groups must contain at least two groups"
            assert (
                len(layer_groups) < low_dim
            ), "low_dim must be greater than the number of layer groups"
            for group in layer_groups:
                assert isinstance(group, list), "layer_groups must be a list of lists"

        # Hide this from inspection by get_parameters()
        model = deepcopy(model)
        self.model = [model]

        # Data attributes of the class
        self.low_dim = low_dim
        self.param_replacements = [
            ParamReplacement(model, param, param_name, layer_groups)
            for param_name, param in dict(model.named_parameters()).items()
            if param.requires_grad
        ]

        # Parameter vector that is updated
        # Initialized with zeros as per text: \theta^{d}
        num_layer_groups = len(layer_groups) if layer_groups is not None else 0
        self.low_dim_params = nn.Parameter(torch.zeros(low_dim - num_layer_groups))

        # Scaler for different layer groups
        self.group_scalers: nn.Parameter | None = None
        if layer_groups is not None:
            self.group_scalers = nn.Parameter(torch.ones(num_layer_groups))

    def forward(self, x: Any) -> Any:
        """
        Set the parameters of the model using the FastFood transform
        if we're training so that we can get gradients on self.low_dim_params,
        then call the model with the updated parameters.

        Args:
            x (Any): Original model's input.

        Returns:
            Any: Original model's output.
        """
        if self.training:
            self.set_params()

        # Pass through the model, by getting the module from a list self.model
        model = self.model[0]
        x = model(x)
        return x

    def set_params(self) -> None:
        for replacement in self.param_replacements:
            replacement.set_param(self.low_dim_params, self.group_scalers)


class ParamReplacement(nn.Module):
    """
    A helper class to replace a parameter of a model
    with those computed by the Fastfood transform.
    """

    def __init__(
        self,
        base_module: nn.Module,
        param: nn.Parameter,
        param_name: str,
        layer_groups: list[list[str]] | None = None,
    ) -> None:
        """
        Args:
            base_module (nn.Module): The model that contains the parameter to replace.
            param (nn.Parameter): The parameter to replace.
            param_name (str): The name of the parameter to replace in the model's attributes.
            layer_groups (list[list[str]] | None, optional): Layers that we'll be replacing
                split by groups, so that we can determine the group to which the current
                parameter belongs and scale its modified parameters accordingly.
                Defaults to None, in which case no group scalers will be used.
        """
        super().__init__()
        assert param.requires_grad

        # Saves the initial values of the initialized parameters
        # from param.data and sets them to no grad.
        # (initial values are the 'origin' of the search)
        initial_value = param.clone().detach()

        # Create the Fastfood transform matrices
        param_dim = np.prod(param.size())
        fastfood_vars = make_fastfood_vars(param_dim)

        # Get the base nn.Module (i.e. the layer of the parameter),
        # and the local name of the parameter withing that module.
        # e.g. base_module = nn.Conv2d(...), param_name = "weight"
        param_local_name = param_name
        while "." in param_local_name:
            prefix, param_local_name = param_local_name.split(".", 1)
            base_module = base_module.__getattr__(prefix)

        # Delete the original parameter from the base nn.Module
        # so that we can replace them with a tensor, which
        # we'll initialize to the original parameter's value.
        delattr(base_module, param_local_name)
        setattr(base_module, param_local_name, initial_value)

        # Get the group index for this parameter
        group: int | None = None
        if layer_groups is not None:
            layer_to_group = {
                layer: group_idx
                for group_idx, layer_group in enumerate(layer_groups)
                for layer in layer_group
            }
            for layer, group_idx in layer_to_group.items():
                if param_name.startswith(layer):
                    group = group_idx
                    break
            assert (
                group is not None
            ), f"Layer containing {param_name} not found in layer_groups"

        # Attributes required for Fastfood transform
        self.register_buffer("initial_value", initial_value)
        self.param_dim = param_dim
        for name, var in fastfood_vars.items():
            if isinstance(var, torch.Tensor):
                self.register_buffer("fastfood_" + name, var)
        self.fastfood_LL = fastfood_vars["LL"]

        # Attributes required to swap out module parameters
        # with Fastfood transform projections
        self.base_module = base_module
        self.param_name = param_name
        self.param_local_name = param_local_name
        self.group = group

    def set_param(
        self, low_dim_params: nn.Parameter, group_scalers: nn.Parameter | None = None
    ) -> None:
        """
        Replace the parameters of the base model with the Fastfood transform projections.

        Args:
            low_dim_params (nn.Parameter): Low-dimensional parameters that are projected
                to the original parameter's size using the Fastfood transform.
            group_scalers (nn.Parameter | None, optional): The scalers used for each
                layer group. Defaults to None, in which case the projections are not scaled.
        """
        # Get the Fastfood transform delta to add to the initial value
        fastfood_var_dict = {
            "BB": self.fastfood_BB,
            "Pi": self.fastfood_Pi,
            "GG": self.fastfood_GG,
            "divisor": self.fastfood_divisor,
            "LL": self.fastfood_LL,
        }
        delta = fastfood_torched(low_dim_params, self.param_dim, fastfood_var_dict)
        delta = delta.view(self.initial_value.size())

        # Scale the delta by this parameter's group scaler
        if group_scalers is not None:
            assert self.group is not None
            scaler = group_scalers[self.group]
            delta = delta * scaler

        param = self.initial_value + delta
        setattr(self.base_module, self.param_local_name, param)


def fast_walsh_hadamard_torched(
    x: torch.Tensor, axis: int = 0, normalize: bool = False
) -> torch.Tensor:
    """
    Performs fast Walsh Hadamard transform

    Args:
        x (torch.Tensor): Input matrix.
        axis (int, optional): Axis along which to perform the transform. Defaults to 0.
        normalize (bool, optional): Whether to normalize the output. Defaults to False.

    Returns:
        torch.Tensor: The output matrix of the transform.
    """
    orig_shape = x.size()
    assert axis >= 0 and axis < len(
        orig_shape
    ), "For a vector of shape %s, axis must be in [0, %d] but it is %d" % (
        orig_shape,
        len(orig_shape) - 1,
        axis,
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2**h_dim_exp, (
        "hadamard can only be computed over axis with size that is a "
        f"power of two, but chosen axis {axis} has size {h_dim}"
    )

    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [
        int(np.prod(orig_shape[axis + 1 :]))
    ]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrays = torch.chunk(ret, 2, dim=dim)
        assert len(arrays) == 2
        ret = torch.cat((arrays[0] + arrays[1], arrays[0] - arrays[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def make_fastfood_vars(dim: int) -> dict[str, torch.Tensor]:
    """
    Creates variables for Fastfood transform
    (an efficient random projection, see: https://arxiv.org/abs/1804.08838)

    Args:
        dim (int): Projection size of the output matrix.

    Returns:
        dict[str, torch.Tensor]: A dictionary of variables for the transform.
    """
    ll = int(np.ceil(np.log(dim) / np.log(2)))
    LL = 2**ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1).type(torch.FloatTensor)
    BB.requires_grad = False

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL))
    Pi.requires_grad = False

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(
        LL,
    ).normal_()
    GG.requires_grad = False

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    return {"BB": BB, "Pi": Pi, "GG": GG, "divisor": divisor, "LL": LL}


def fastfood_torched(
    x: torch.Tensor, dim: int, param_dict: dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Fastfood transform

    Args:
        x (torch.Tensor): Matrix that we want to project to 'dim' dimensions.
        dim int: Dimensionality of the matrix we want to project to.
        param_dict (dict[str, torch.Tensor]): Dictionary of Fastfood
            transform variables.

    Returns:
        torch.Tensor: The Fastfood transform of the input matrix.
    """
    dim_source = x.size(0)

    # Pad x if needed
    dd_pad = F.pad(x, pad=(0, param_dict["LL"] - dim_source), value=0, mode="constant")

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(param_dict["BB"], dd_pad)
    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[param_dict["Pi"]]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, param_dict["GG"])

    # (HGPiHBX)
    mul_5 = fast_walsh_hadamard_torched(mul_4, 0, normalize=False)

    ret = torch.div(
        mul_5[:dim],
        param_dict["divisor"] * np.sqrt(float(dim) / param_dict["LL"]),
    )

    return ret
