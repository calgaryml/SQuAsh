// Copyright (c) 2023-2024, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

// Copyright (c) 2024, University of Calgary, developed by Mike Lasby & Mohamed Yassin
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#pragma once
using torch::Tensor;
using OptTensor = torch::optional<Tensor>;

template <typename T>
using VectorData = Kokkos::mdspan<T, Kokkos::dextents<std::int64_t, 1>, Kokkos::layout_right>;

template <typename T>
using MatrixData = Kokkos::mdspan<T, Kokkos::dextents<std::int64_t, 2>, Kokkos::layout_right>;

template <class T>
auto data_ptr(const Tensor &data)
{
    return data.data_ptr<T>();
}

template <>
auto data_ptr<std::uint16_t>(const Tensor &data)
{
    return reinterpret_cast<std::uint16_t *>(data.data_ptr<std::int16_t>());
}

template <class T>
MatrixData<T> to_matrix(const Tensor &data)
{
    TORCH_CHECK(data.is_contiguous())
    TORCH_CHECK_EQ(data.dim(), 2);
    return Kokkos::mdspan(data_ptr<T>(data), data.size(0), data.size(1));
}

template <class T>
auto to_matrix(const OptTensor &data) -> decltype(std::make_optional(to_matrix<T>(data.value())))
{
    if (data.has_value())
    {
        return std::make_optional(to_matrix<T>(data.value()));
    }
    else
    {
        return std::nullopt;
    }
}

template <class T>
VectorData<T> to_vector(const Tensor &data)
{
    TORCH_CHECK(data.is_contiguous())
    TORCH_CHECK_EQ(data.dim(), 1);
    return Kokkos::mdspan(data.data_ptr<T>(), data.size(0));
}

template <class T>
auto to_vector(const OptTensor &data) -> decltype(std::make_optional(to_vector<T>(data.value())))
{
    if (data.has_value())
    {
        return std::make_optional(to_vector<T>(data.value()));
    }
    else
    {
        return std::nullopt;
    }
}

torch::Tensor allocate_output(const torch::Tensor &ref, std::int64_t dim0, std::int64_t dim1)
{
    return at::empty({dim0, dim1}, at::device(ref.device()).dtype(ref.dtype()));
}