import torch
import torch.nn as nn
import torch.functional as F

from mlp_hip_train import FFI

class FFILinear(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        dtype: torch.typename = torch.float32,
        transpose: bool = True,
        vectorize: bool = True,
        index_dtype: torch.typename = torch.int32,
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype

        self.transpose = transpose
        with torch.no_grad():
            active_neuron_idx = module.weight.sum(dim=1) != 0
            fine_grained_idx = (module.weight[active_neuron_idx] != 0).to(
                torch.bool
            )
            _, self.input_mask = fine_grained_idx.nonzero(as_tuple=True)
            self.input_mask = self.input_mask.reshape(
                shape=(module.weight[active_neuron_idx].shape[0], -1)
            ).to(index_dtype)
            weight = module.weight[active_neuron_idx].detach().type(dtype)
            weight = torch.clone(
                weight[fine_grained_idx]
                .reshape(shape=(weight.shape[0], -1))
                .detach()
                .type(dtype)
            )
            # padding to multiple of 4
            if vectorize:
                pad = (
                    self.input_mask.shape[1] + 3
                ) // 4 * 4 - self.input_mask.shape[1]
                self.input_mask = F.pad(self.input_mask, [0, pad])
                weight = F.pad(weight, [0, pad])

            self.condensed_weight = nn.Parameter(
                weight,
                requires_grad=False,
            )

            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(
                        module.bias[active_neuron_idx].detach().type(dtype)
                    ),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return FFI.ffi_mul(
            input,
            self.condensed_weight,
            self.input_mask,
            self.bias,
            transpose=self.transpose,
        )