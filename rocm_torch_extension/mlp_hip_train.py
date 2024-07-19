import math
from typing import Any, Optional
from torch import nn
from torch.autograd import Function
import torch

import mlp_hip

class FFIMulFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx: Any,
                features: torch.Tensor,
                weights: torch.Tensor,
                locations: torch.Tensor,
                bias: Optional[torch.Tensor],
                transpose: bool = True) -> torch.Tensor:
        ctx.transpose = transpose

        # if transpose:
        #     features_tp = torch.transpose(features, 0, 1).contiguous()
        #     ctx.save_for_backward(features_tp, weights, locations)
        #     result = mlp_hip.ffi_forward_tp(features_tp, weights, locations, bias)
        # else:
        features = features.contiguous()
        ctx.save_for_backward(features, weights, locations)
        result = mlp_hip.ffi_forward(features, weights, locations, bias)


        if bias is not None:
            ctx.use_bias = True
        else:
            ctx.use_bias = False
        return result

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        features, weights, locations, = ctx.saved_tensors
        transpose = ctx.transpose

        grad_output = grad_output.contiguous()

        if ctx.use_bias:
            bias_gradient = torch.sum(grad_output, dim=0).detach()
        else:
            bias_gradient = None

        # if transpose:
        #     input_gradient, weight_gradient, = mlp_hip.ffi_backward_tp(
        #         features, weights, locations, grad_output
        #     )
        #     input_gradient = torch.transpose(input_gradient, 0, 1)
        # else:
        input_gradient, weight_gradient, = mlp_hip.ffi_backward(
            features, weights, locations, grad_output
        )

        return input_gradient, weight_gradient, None, bias_gradient, None


class FFI:
    def ffi_mul(features: torch.Tensor,
                weights: torch.Tensor,
                locations: torch.Tensor,
                bias: Optional[torch.Tensor], 
                transpose: bool = True):
        """
    Performs a fixed fan-in batched matrix multiplication.
    Multiplies `features` with the fixed fan-in sparse matrix described by `weights` and `locations`, possibly
    adding a `bias` term.
    :param features: Input features, floating-point tensor of shape `batch_size x num_features`
    :param weights: Weight values, floating-point tensor of shape `num_outputs x fan_in`
    :param locations: Weight locations, integer tensor of shape `num_outputs x fan_in`
    :param bias: Optional bias term. Floating-point tensor of shape `num_outputs`.
    :param transpose: Whether to transpose the input before multiplying. Generally recommended for CUDA to improve
    performance, unless batch sizes are very small.
    :return: A floating-point tensor of shape `batch_size x num_outputs`.
    """
        if features.ndim > 2:
            original_shape = features.shape[:-1]
            reshaped = features.view(-1, features.shape[-1])
            result = FFIMulFunction.apply(reshaped, weights, locations, bias, transpose)
            return result.view(original_shape + (locations.shape[0],))
        else:
            return FFIMulFunction.apply(features, weights, locations, bias, transpose)