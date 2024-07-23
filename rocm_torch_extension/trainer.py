from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch
from mlp_hip_train import FFI

device = torch.device("cuda")

print('Using device:', device)

# Test #1 
batch_size = 30
seq_len = 6
feature_size = 165
output_size = 503
fan_in = 24

features = torch.randn(batch_size, seq_len, feature_size, dtype=torch.float32, requires_grad=True, device=device)
weights = torch.randn(output_size, fan_in, dtype=torch.float32, requires_grad=True, device=device)
locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int32, requires_grad=False,
                            device=device)

transpose = True

# do the multiplication jointly
all_results = FFI.ffi_mul(features, weights, locations, None, transpose)
assert all_results.shape == (batch_size, seq_len, output_size)
print(all_results)

# and compare result with slices
one_result = FFI.ffi_mul(features[:, 5, :], weights, locations, None, transpose)
assert (all_results[:, 5, :] == one_result).all()

one_result = FFI.ffi_mul(features[8, :, :], weights, locations, None, transpose)
assert (all_results[8, :, :] == one_result).all()

# Test int16 and int32 indicies give the same result
locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int16, requires_grad=False,
                            device=device)
int16_result = FFI.ffi_mul(features, weights, locations.to(torch.int16), None, transpose)
int32_result = FFI.ffi_mul(features, weights, locations.to(torch.int32), None, transpose)
assert (int16_result == int32_result).all()
