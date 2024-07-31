# Copyright (c) 2023-2024, Aalto University, developed by Erik Schultheis
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024, University of Calgary, developed by Mike Lasby & Mohamed Yassin
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import division
from __future__ import print_function

import torch
from mlp_hip_train import FFI
import torch.nn as nn
from torch.autograd import gradcheck


from mlp_hip_train import FFI
from squash.ffi_linear import FFILinear
from squash.utils import generate_ffi_structure, assert_ffi

def main():
    device = torch.device("cuda")

    print('Using device:', device)

    # Test #1 
    batch_size = 36
    seq_len = 12
    feature_size = 256
    output_size = 512
    fan_in = 24

    features = torch.randn(batch_size, seq_len, feature_size, dtype=torch.float32, requires_grad=True, device=device)
    weights = torch.randn(output_size, fan_in, dtype=torch.float32, requires_grad=True, device=device)
    locations = torch.randint(0, feature_size, (output_size, fan_in), dtype=torch.int32, requires_grad=False,
                                device=device)

    transpose = True

    # do the multiplication jointly
    all_results = FFI.ffi_mul(features, weights, locations, None, transpose)
    assert all_results.shape == (batch_size, seq_len, output_size)
    # print(all_results)

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

    # compare dense to sparse
    _sparsity = 1- (fan_in / feature_size)
    dense = nn.Linear(in_features=feature_size, out_features=output_size, device=device)
    sparse = nn.Linear(in_features=feature_size, out_features=output_size, device=device)
    sparse.weight.data = dense.weight.detach().clone()
    sparse.bias.data = dense.bias.detach().clone()
    sparse_ffi_weight = generate_ffi_structure(t = sparse.weight, sparsity=_sparsity)
    sparse.weight.data = sparse_ffi_weight
    assert_ffi(sparse.weight)
    ffi_linear = FFILinear(sparse, transpose=True, vectorize=True)
    dense_out = sparse(features)
    sparse_out = ffi_linear(features)
    assert torch.allclose(dense_out, sparse_out, atol=1e-7)

    # grad check
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
    if bias:
        bias = torch.randn(output_size, dtype=torch.float32, requires_grad=True, device=device)
    else:
        bias = None
    test = gradcheck(FFI.ffi_mul, (features, weights, locations, bias, transpose), eps=1e-4, atol=1e-5, rtol=1e-3,
                        fast_mode=True, nondet_tol=1e-5)
        

if __name__ == "__main__":
    main()
