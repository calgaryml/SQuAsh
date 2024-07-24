from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch
from mlp_hip_train import FFI
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")
torch.manual_seed(42)
print('Using device:', device)

# Test #1 

# batch_size = 36
# seq_len = 12
# feature_size = 256
# output_size = 512
# fan_in = 24

# features = torch.randn(batch_size, seq_len, feature_size, dtype=torch.float32, requires_grad=True, device=device)
# weights = torch.randn(output_size, fan_in, dtype=torch.float32, requires_grad=True, device=device)
# locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int32, requires_grad=False,
#                             device=device)

# transpose = False

# # do the multiplication jointly
# all_results = FFI.ffi_mul(features, weights, locations, None, transpose)
# assert all_results.shape == (batch_size, seq_len, output_size)
# torch.set_printoptions(threshold=100_000)
# # print(all_results)

# # and compare result with slices
# one_result = FFI.ffi_mul(features[:, 5, :], weights, locations, None, transpose)
# assert (all_results[:, 5, :] == one_result).all()

# one_result = FFI.ffi_mul(features[8, :, :], weights, locations, None, transpose)
# assert (all_results[8, :, :] == one_result).all()

# # Test int16 and int32 indicies give the same result
# locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int16, requires_grad=False,
#                             device=device)
# int16_result = FFI.ffi_mul(features, weights, locations.to(torch.int16), None, transpose)
# int32_result = FFI.ffi_mul(features, weights, locations.to(torch.int32), None, transpose)
# assert (int16_result == int32_result).all()


# batch_size = 36
# seq_len = 12
# feature_size = 256
# output_size = 512
# fan_in = 24

# features = torch.randn(batch_size, seq_len, feature_size, dtype=torch.float32, requires_grad=True, device=device)
# weights = torch.randn(output_size, fan_in, dtype=torch.float32, requires_grad=True, device=device)
# locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int32, requires_grad=False,
#                             device=device)
# out_grad = torch.randn(batch_size, output_size, dtype=torch.float32, requires_grad=True, device=device)
# mlp_hip.ffi_backward_tp(features, weights, locations, out_grad)


batch_size = 36
seq_len = 12
feature_size = 256
output_size = 512
fan_in = 24
bias = False
transpose = True
features = torch.randn(batch_size, feature_size, dtype=torch.float32, requires_grad=True, device=device)
weights = torch.randn(output_size, fan_in, dtype=torch.float32, requires_grad=True, device=device)
locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int32, requires_grad=False,
                            device=device)

transpose = False

# do the multiplication jointly
all_results = FFI.ffi_mul(features, weights, locations, None, transpose)
assert all_results.shape == (batch_size, seq_len, output_size)
torch.set_printoptions(threshold=100_000)
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
