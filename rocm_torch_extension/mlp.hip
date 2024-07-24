#include <torch/extension.h>
#include <vector>
#include <hip/hip_runtime.h>
#include <type_traits>
#include "forward_data.hpp"
#include "backward_data.hpp"
#include "enums.hpp"
#include "generic_vector.hip"

using torch::Tensor;
using OptTensor = torch::optional<Tensor>;
#define HIP_ASSERT(x) (assert((x) == hipSuccess));
#define XCK_PRAGMA_UNROLL _Pragma("unroll")

template <class T>
struct mixed_precision_selector;

// 32-bit float accumulates in 32 bit
template <>
struct mixed_precision_selector<float>
{
  using accumulator = float;
};

// 64-bit float accumulates in 64 bit
template <>
struct mixed_precision_selector<double>
{
  using accumulator = double;
};

template <class T>
using mixed_precision_accumulator_t = typename mixed_precision_selector<T>::accumulator;

/// Helper template to generate a `std::integral_constant` type of the given value.
template <auto Value>
using make_const_t = std::integral_constant<decltype(Value), Value>;

//! Integer division with rounding up.
template <class T, class X = std::enable_if_t<std::is_integral_v<T>>>
constexpr inline T div_up(T value, T divisor)
{
  return (value + (divisor - 1)) / divisor;
}

//! Rounding `value` up to the next multiple of `divisor`
template <class T, class X = std::enable_if_t<std::is_integral_v<T>>>
constexpr inline T round_up(T value, T divisor)
{
  return div_up(value, divisor) * divisor;
}

template <typename Float, typename Int, bool Transpose>
struct ForwardKernelHelpers
{
  using ForwardKernelData = ForwardData<Float, Int, Transpose>;
  using int_vec_t = typename ForwardKernelData::int_vec_t;
  using float_vec_t = typename ForwardKernelData::float_vec_t;
  using accumulate_t = mixed_precision_accumulator_t<Float>;

  static __device__ accumulate_t loopFan(ForwardKernelData &data, std::int32_t batchIndex, std::int32_t unitIndex)
  {
    assert(0 <= batchIndex < data.batchSize_);
    assert(0 <= unitIndex < data.outputSize_);

    accumulate_t value = data.bias_at(unitIndex);

    for (std::int32_t fanIndex = 0; fanIndex < data.lookupSize_; fanIndex += 4)
    {
      int_vec_t indices = data.lookup_vec_at(unitIndex, fanIndex);
      float_vec_t weights = data.weight_vec_at(unitIndex, fanIndex);

      XCK_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j)
      {
        Float feature = data.feature_at(batchIndex, indices[j]);
        value += accumulate_t(feature) * accumulate_t(weights[j]);
      }
    }
    return value;
  }

  template <EGrid2dMode Mode>
  static __device__ void grid2d(ForwardKernelData &data, std::integral_constant<EGrid2dMode, Mode> mode, std::int32_t unitOffset = 0)
  {
    std::int32_t batchIndex, unitIndex;
    if constexpr (Mode == EGrid2dMode::BATCH_UNIT_TG)
    {
      batchIndex = blockIdx.y * blockDim.x + threadIdx.x;
      unitIndex = blockIdx.x * blockDim.y + threadIdx.y;
    }
    else if (Mode == EGrid2dMode::BATCH_UNIT)
    {
      batchIndex = blockIdx.x * blockDim.x + threadIdx.x;
      unitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    }
    unitIndex += unitOffset;
    if (batchIndex >= data.batchSize_ || unitIndex >= data.outputSize_)
      return;
    data.output_at(batchIndex, unitIndex) = loopFan(data, batchIndex, unitIndex);
  };
};

template <typename Float, typename Int>
struct BackwardKernelHelpers
{
  using BackwardKernelData = BackwardData<Float, Int>;
  static __device__ void loop_fan_vector(BackwardKernelData &data,
                                         std::int32_t batch_index,
                                         std::int32_t unit_index)
  {
    assert(0 <= batch_index < data.batchSize_);
    assert(0 <= unit_index < data.outputSize_);

    Float out = data.output_at(batch_index, unit_index);
    if (out == Float{0})
    {
      return;
    }

    for (std::int32_t weight_index = 0; weight_index < data.lookupSize_; weight_index += 4)
    {
      auto source4 = data.lookup_vec_at(unit_index, weight_index);
      auto weight4 = data.weight_vec_at(unit_index, weight_index);
      XCK_PRAGMA_UNROLL
      for (int k = 0; k < 4; ++k)
      {
        Float increment = weight4[k] * out;
        atomicAdd(&data.feature_at(batch_index, source4[k]), increment);
      }
    }
  }
  static __device__ void grid_batch_unit(BackwardKernelData &data)
  {
    std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
    std::int32_t unit_block_size = blockDim.y * gridDim.y;
    std::int32_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_index >= data.batchSize_)
    {
      return;
    }

    for (std::int32_t unit_index = unit_index_offset; unit_index < data.outputSize_; unit_index += unit_block_size)
    {
      loop_fan_vector(data, batch_index, unit_index);
    }
  }

  static __device__ void loop_batch_scalar(BackwardKernelData &data,
                                           std::int32_t unit_index,
                                           std::int32_t weight_index)
  {

    assert(0 <= weight_index < data.lookupSize_);
    assert(0 <= unit_index < data.outputSize_);

    Int source = data.lookup_at(unit_index, weight_index);
    Float weight = data.weight_at(unit_index, weight_index);
    for (std::int32_t batch_index = 0; batch_index < data.batchSize_; ++batch_index)
    {
      Float out = data.output_at(batch_index, unit_index);
      Float increment = weight * out;
      atomicAdd(&data.feature_at(batch_index, source), increment);
    }
  }

  static __device__ void grid_fan_unit(BackwardKernelData &data)
  {
    std::int32_t weight_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (weight_index >= data.lookupSize_)
    {
      return;
    }

    std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
    std::int32_t unit_block_size = blockDim.y * gridDim.y;

    for (std::int32_t unit_index = unit_index_offset; unit_index < data.outputSize_; unit_index += unit_block_size)
    {
      loop_batch_scalar(data, unit_index, weight_index);
    }
  }
};

namespace kernels
{
  template <typename Float, typename Int, bool Transpose>
  __global__ void ffi_bu_f(ForwardData<Float, Int, Transpose> data, std::int32_t unit_offset)
  {
    ForwardKernelHelpers<Float, Int, Transpose>::grid2d(data, make_const_t<EGrid2dMode::BATCH_UNIT>{}, unit_offset);
  }

  template <typename Float, typename Int, bool Transpose>
  __global__ void ffi_bu_tg_f(ForwardData<Float, Int, Transpose> data)
  {
    ForwardKernelHelpers<Float, Int, Transpose>::grid2d(data, make_const_t<EGrid2dMode::BATCH_UNIT_TG>{});
  }

  template <typename Float, typename Int>
  __global__ void __launch_bounds__(1024)
      ffi_bf_bu_f_vector(BackwardData<Float, Int> data)
  {
    BackwardKernelHelpers<Float, Int>::grid_batch_unit(data);
  }

  template <typename Float, typename Int>
  __global__ void __launch_bounds__(1024)
      ffi_bf_fu_b_scalar(BackwardData<Float, Int> data)
  {
    BackwardKernelHelpers<Float, Int>::grid_fan_unit(data);
  }

}

struct sDim3d
{
  std::uint32_t x = 1;
  std::uint32_t y = 1;
  std::uint32_t z = 1;
};

namespace ForwardPass
{
  template <typename Float, typename Int, bool Transpose>
  void call(std::bool_constant<Transpose> transpose [[maybe_unused]],
            ForwardImplementations strategy,
            MatrixData<Float> features,
            MatrixData<Int> lookup,
            MatrixData<Float> weights,
            std::optional<VectorData<Float>> biases,
            MatrixData<Float> output)
  {
    using OutT = ForwardData<Float, Int, Transpose>;
    auto data = OutT::from_matrix(features, lookup, weights, biases, output);

    dim3 block_final = dim3{32, 16};
    if (data.batchSize_ <= 4 || (9 <= data.batchSize_ && data.batchSize_ <= 12))
    {
      block_final = {4, 96};
    }
    else if (data.batchSize_ <= 8 || (17 <= data.batchSize_ && data.batchSize_ <= 24))
    {
      block_final = {8, 48};
    }
    else if (data.batchSize_ <= 16)
    {
      block_final = {16, 24};
    }
    sDim3d grid;
    sDim3d grid_dim;
    if (transpose)
    {
      grid = {(std::uint32_t)data.outputSize_, (std::uint32_t)data.batchSize_};
      grid_dim = {
          div_up(grid.x, block_final.y),
          div_up(grid.y, block_final.x),
          div_up(grid.z, block_final.z)};
    }
    else
    {
      grid = {(std::uint32_t)data.batchSize_, (std::uint32_t)data.outputSize_};
      grid_dim = {
          div_up(grid.x, block_final.x),
          div_up(grid.y, block_final.y),
          div_up(grid.z, block_final.z)};
    }
    if (grid_dim.y > 65'535)
    {
      grid_dim.y = 65'535;
    }
    dim3 grid_final = dim3{grid_dim.x, grid_dim.y, grid_dim.z};
    // hipStream_t Stream = nullptr; //Check streams later
    if (strategy == ForwardImplementations::GPU_BatchUnit_TG_Fan)
    {
      kernels::ffi_bu_tg_f<<<grid_final, block_final>>>(data);
      HIP_ASSERT(hipDeviceSynchronize());
    }
    else if (strategy == ForwardImplementations::GPU_BatchUnit_Fan)
    {
      for (std::int32_t unit_pos = 0; unit_pos < data.outputSize_; unit_pos += grid_final.y * block_final.y)
      {
        {
          kernels::ffi_bu_f<<<grid_final, block_final>>>(data, unit_pos);
          HIP_ASSERT(hipDeviceSynchronize());
        }
      }
    }
  }
}

namespace BackwardPass
{
  namespace FtrOP
  {
    template <typename Float, typename Int, bool Transpose>
    void call(ForwardImplementations strategy,
              MatrixData<Float> features,
              MatrixData<Int> lookup,
              MatrixData<Float> weights,
              std::optional<VectorData<Float>> biases,
              MatrixData<Float> output)
    {
      using OutT = ForwardData<Float, Int, Transpose>;
      auto data = OutT::from_matrix(features, lookup, weights, biases, output);

      int device;
      HIP_ASSERT(hipGetDevice(&device));
      hipDeviceProp_t properties;
      HIP_ASSERT(hipGetDeviceProperties(&properties, device));
      // calculate how many threads (in multiples of warps) we get per multiprocessor if we distribute evenly

      dim3 block_final = dim3{32, 16};
      if (data.batchSize_ <= 4 || (9 <= data.batchSize_ && data.batchSize_ <= 12))
      {
        block_final = {4, 96};
      }
      else if (data.batchSize_ <= 8 || (17 <= data.batchSize_ && data.batchSize_ <= 24))
      {
        block_final = {8, 48};
      }
      else if (data.batchSize_ <= 16)
      {
        block_final = {16, 24};
      }
      sDim3d grid;
      sDim3d grid_dim;
      if (transpose)
      {
        grid = {(std::uint32_t)data.outputSize_, (std::uint32_t)data.batchSize_};
        grid_dim = {
            div_up(grid.x, block_final.y),
            div_up(grid.y, block_final.x),
            div_up(grid.z, block_final.z)};
      }
      else
      {
        grid = {(std::uint32_t)data.batchSize_, (std::uint32_t)data.outputSize_};
        grid_dim = {
            div_up(grid.x, block_final.x),
            div_up(grid.y, block_final.y),
            div_up(grid.z, block_final.z)};
      }
      if (grid_dim.y > 65'535)
      {
        grid_dim.y = 65'535;
      }
      dim3 grid_final = dim3{grid_dim.x, grid_dim.y, grid_dim.z};
      // hipStream_t Stream = nullptr; //Check streams later
      if (strategy == ForwardImplementations::GPU_BatchUnit_TG_Fan)
      {
        kernels::ffi_bu_tg_f<<<grid_final, block_final>>>(data);
      }
      else if (strategy == ForwardImplementations::GPU_BatchUnit_Fan)
      {
        for (std::int32_t unit_pos = 0; unit_pos < data.outputSize_; unit_pos += grid_final.y * block_final.y)
        {
          {
            kernels::ffi_bu_f<<<grid_final, block_final>>>(data, unit_pos);
            HIP_ASSERT(hipDeviceSynchronize());
          }
        }
      }
    }
  }
  namespace WgtOP
  {

  }
}

template <class Float, bool Transpose>
struct SparseForwardOp
{
  template <class Int>
  static Tensor run(
      const Tensor &features,
      const Tensor &weights,
      const Tensor &locations,
      const OptTensor &bias_opt)
  {
    ForwardImplementations strategy;
    if (Transpose)
      strategy = ForwardImplementations::GPU_BatchUnit_TG_Fan;
    else
    {
      strategy = ForwardImplementations::GPU_BatchUnit_Fan;
    }
    TORCH_CHECK_VALUE(weights.size(0) == locations.size(0), "mismatch in number of outputs")
    TORCH_CHECK_VALUE(weights.size(1) == locations.size(1), "mismatch in fan-in")
    if (bias_opt.has_value())
    {
      TORCH_CHECK_VALUE(weights.size(0) == bias_opt.value().size(0), "mismatch in number of outputs")
    }

    std::int64_t batch_size = Transpose ? features.size(1) : features.size(0);
    std::int64_t output_size = weights.size(0);

    Tensor output = allocate_output(features, batch_size, output_size);

    ForwardPass::call(
        std::bool_constant<Transpose>{},
        strategy,
        to_matrix<Float>(features),
        to_matrix<Int>(locations),
        to_matrix<Float>(weights),
        to_vector<Float>(bias_opt),
        to_matrix<Float>(output));
    return output;
  }

  static Tensor dispatch(
      const Tensor &input,
      const Tensor &weights,
      const Tensor &locations,
      const OptTensor &bias_opt)
  {
    if (locations.dtype() == torch::kInt32)
    {
      return run<std::int32_t>(input, weights, locations, bias_opt);
    }
    else if (locations.dtype() == torch::kInt16)
    {
      return run<std::uint16_t>(input, weights, locations, bias_opt);
    }
    else
    {
      TORCH_CHECK_TYPE(false, "locations must be int32 or int16")
    }
  }
};

template <class Float>
struct SparseBackwardOp
{
  template <class Int>
  static std::vector<Tensor> run(
      const Tensor &features,
      const Tensor &weights,
      const Tensor &locations,
      const Tensor &output_gradient)
  {
    BackwardFtrImplementations strategy_ftr = BackwardFtrImplementations::GPU_BatchUnit_Fan;
    BackwardWgtImplementations strategy_wgt = BackwardWgtImplementations::GPU_FanUnit_Batch;

    std::int64_t batch_size = features.size(1);

    TORCH_CHECK_VALUE(weights.size(0) == locations.size(0), "mismatch in number of outputs")
    TORCH_CHECK_VALUE(weights.size(1) == locations.size(1), "mismatch in fan-in")
    TORCH_CHECK_VALUE(output_gradient.size(0) == batch_size, "mismatch in batch size ", output_gradient.size(0), " vs ", batch_size)
    TORCH_CHECK_VALUE(output_gradient.size(1) == weights.size(0), "mismatch in number of outputs");

    Tensor input_gradient = allocate_output(features, features.size(0), features.size(1));
    Tensor weight_gradient = allocate_output(weights, weights.size(0), weights.size(1));

    BackwardPass::FtrOP::call(
        strategy,
        to_matrix<Int>(locations),
        to_matrix<Float>(weights),
        to_matrix<Float>(output_gradient),
        to_matrix<Float>(input_gradient));
    // BackwardPass::WgtOP::call(
    //     strategy,
    //     to_matrix<Float>(input_gradient),
    //     to_matrix<Int>(locations),
    //     to_matrix<Float>(output_gradient),
    //     to_matrix<Float>(weights));
    return {input_gradient, weight_gradient};
  }

  static std::vector<Tensor> dispatch(
      const Tensor &input,
      const Tensor &weights,
      const Tensor &locations,
      const Tensor &output_gradient)
  {
    if (locations.dtype() == torch::kInt32)
    {
      return run<std::int32_t>(input, weights, locations, output_gradient);
    }
    else if (locations.dtype() == torch::kInt16)
    {
      return run<std::uint16_t>(input, weights, locations, output_gradient);
    }
    else
    {
      TORCH_CHECK_TYPE(false, "locations must be int32 or int16")
    }
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("ffi_forward", &SparseForwardOp<float, false>::dispatch, "Performs a fixed fan-in batched matrix multiplication.");
  m.def("ffi_forward_tp", &SparseForwardOp<float, true>::dispatch, "Performs a fixed fan-in batched matrix multiplication.");
  m.def("ffi_backward_tp", &SparseBackwardOp<float>::dispatch, "Performs a fixed fan-in batched matrix multiplication.");
  m.def("ffi_backward_tp", &SparseBackwardOp<float>::dispatch, "Performs a fixed fan-in batched matrix multiplication.");
}
