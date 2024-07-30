#pragma once
#include "mdspan/include/mdspan/mdspan.hpp"
#include "generic_vector.hip"
#include <type_traits>
#include "data_helpers.hpp"

using torch::Tensor;

template <typename Float, typename Int>
class BackwardData
{
public:
    BackwardData(Float *__restrict__ features, Int *__restrict__ lookup, Float *__restrict__ weights,
                 Float *__restrict__ outputs, std::int64_t batchSize, std::int64_t inputSize,
                 std::int64_t outputSize, std::int64_t lookupSize) : features_(features), lookup_(lookup), weights_(weights),
                                                                     outputs_(outputs), batchSize_(batchSize), inputSize_(inputSize), outputSize_(outputSize),
                                                                     lookupSize_(lookupSize) {}

    Float *__restrict__ features_ = nullptr;
    Int *__restrict__ lookup_ = nullptr;
    Float *__restrict__ weights_ = nullptr;
    Float *__restrict__ outputs_ = nullptr;
    std::int64_t batchSize_;
    std::int64_t inputSize_;
    std::int64_t outputSize_;
    std::int64_t lookupSize_;

    static BackwardData from_matrix(
        MatrixData<Float> features,
        MatrixData<Int> lookup,
        MatrixData<Float> weights,
        MatrixData<Float> outputs)
    {
        std::int64_t batch_size, input_size;

        batch_size = features.extent(1);
        input_size = features.extent(0);
        // verify compatibility of inputs
        assert(batch_size == outputs.extent(0));
        assert(lookup.extent(0) == outputs.extent(1));
        assert(lookup.extent(0) == weights.extent(0));
        assert(lookup.extent(1) == weights.extent(1));

        return BackwardData(features.data_handle(), lookup.data_handle(), weights.data_handle(), outputs.data_handle(),
                            batch_size, input_size, outputs.extent(1), lookup.extent(1));
    }
    using int_vec_t = GenericVector<std::remove_const_t<Int>, 4>;
    using float_vec_t = GenericVector<std::remove_const_t<Float>, 4>;

    __host__ __device__ void *weight_grad_ptr() const { return static_cast<void *>(weights_); }

    __host__ __device__ void *feature_grad_ptr() const { return static_cast<void *>(features_); }

    __device__ Float &feature_at(std::int32_t batch, std::int32_t feature) const
    {
        return features_[feature * batchSize_ + batch];
    }

    __device__ const Int &lookup_at(std::int32_t unit, std::int32_t index) const
    {
        Int &looked_up = lookup_[unit * lookupSize_ + index];
        return looked_up;
    }

    __device__ Float &weight_at(std::int32_t unit, std::int32_t index) const
    {
        return weights_[unit * lookupSize_ + index];
    }

    __device__ Float &output_at(std::int32_t batch, std::int32_t unit)
    {
        return outputs_[batch * outputSize_ + unit];
    }
    __device__ float_vec_t weight_vec_at(std::int32_t unit, std::int32_t index) const
    {
        const Float *base_ptr = weights_ + (unit * lookupSize_ + index);
        return float_vec_t::load(base_ptr);
    }

    __device__ int_vec_t lookup_vec_at(std::int32_t unit, std::int32_t index) const
    {
        const Int *base_ptr = lookup_ + (unit * lookupSize_ + index);
        return int_vec_t::load(base_ptr);
    }
};
