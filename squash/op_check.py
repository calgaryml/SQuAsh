import torch
import unittest
from torch.library import opcheck
from mlp_hip_train import FFI


def sample_inputs():
    device = torch.device("cuda")
    batch_size = 36
    seq_len = 12
    feature_size = 256
    output_size = 512
    fan_in = 24

    features = torch.randn(batch_size, seq_len, feature_size, dtype=torch.float32, requires_grad=True, device=device)
    # features = torch.transpose(features, 0, 1).contiguous()
    weights = torch.randn(output_size, fan_in, dtype=torch.float32, requires_grad=True, device=device)
    locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int32, requires_grad=False,
                                device=device)
    return features, weights, locations

class TestOps(unittest.TestCase):
    def test_ffi_mul(self):
        features, weights, locations = sample_inputs()
        if features.ndim > 2:
            original_shape = features.shape[:-1]
            features = features.view(-1, features.shape[-1])
        transpose=True
        opcheck(torch.ops.mlp_hip.ffi_forward_tp, (features, weights, locations, None))

if __name__ == "__main__":
    unittest.main()