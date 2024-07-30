import math
from typing import Any, Optional
from torch import nn
from torch.autograd import Function
import torch
import pathlib
from torch import _C
module_path = pathlib.Path(__file__).resolve().parent
# library_path = module_path.parent/"build/lib.linux-x86_64-cpython-310/mlp_hip.cpython-310-x86_64-linux-gnu.so"
library_path = module_path.parent/"build/lib.linux-x86_64-3.10/mlp_hip.cpython-310-x86_64-linux-gnu.so"
torch.ops.load_library(library_path)
mlp_hip = torch.ops.mlp_hip

# torch.library.register_autograd(
#     "mlp_hip::ffi_forward_tp", _backward, setup_context=_setup_context)

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

        if transpose:
            features_tp = torch.transpose(features, 0, 1).contiguous()
            ctx.save_for_backward(features_tp, weights, locations)
            result = mlp_hip.ffi_forward_tp(features_tp, weights, locations, bias)
        else:
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

        input_gradient, weight_gradient, = mlp_hip.ffi_backward_tp(
            features, weights, locations, grad_output
        )
        input_gradient = torch.transpose(input_gradient, 0, 1)
        
        return input_gradient, weight_gradient, None, bias_gradient, None


class FFI:
    # @torch.library.custom_op("mlp_hip::ffi_forward_tp", mutates_args=())
    def ffi_mul(features: torch.Tensor,
                weights: torch.Tensor,
                locations: torch.Tensor,
                bias: Optional[torch.Tensor], 
                transpose: bool = True)-> torch.Tensor:
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

    @torch.library.register_fake("mlp_hip::ffi_forward_tp")
    def _(
        features: torch.Tensor,
        weights: torch.Tensor,
        locations: torch.Tensor,
        bias: Optional[torch.Tensor],
        )-> torch.Tensor:
        return torch.empty(size=(features.shape[1], weights.shape[0]), device=torch.device("cuda"))