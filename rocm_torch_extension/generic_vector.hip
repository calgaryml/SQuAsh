// Copyright (c) 2023-2024, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

// Copyright (c) 2024, University of Calgary, developed by Mike Lasby & Mohamed Yassin
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#pragma once
#include <cstring>

template <class ElementType, std::size_t ElementCount>
class alignas(sizeof(ElementType) * ElementCount) GenericVector
{
    static_assert(std::is_trivial_v<ElementType>, "Only trivial types are supported");

public:
    GenericVector() = default;

    constexpr static GenericVector constant(ElementType value)
    {
        GenericVector result;
        for (int k = 0; k < size; ++k)
        {
            result.values[k] = value;
        }
        return result;
    }

    constexpr static GenericVector zeros()
    {
        return constant(0.f);
    }

    constexpr static GenericVector ones()
    {
        return constant(1.f);
    }

    constexpr ElementType &operator[](int index)
    {
        return values[index];
    }

    constexpr const ElementType &operator[](int index) const
    {
        return values[index];
    }

    static constexpr const std::size_t size = ElementCount;
    static constexpr const std::size_t bytes = ElementCount * sizeof(ElementType);

    static __host__ __device__ GenericVector load(const ElementType *address)
    {
        return GenericVector(address);
    }

private:
    explicit __host__ __device__ GenericVector(const ElementType *address)
    {
        if constexpr (bytes % sizeof(int4) == 0)
        {
            const int4 *read_address = reinterpret_cast<const int4 *>(address);
            for (int i = 0; i < bytes; i += sizeof(int4))
            {
                int4 val = *read_address;
                ++read_address;
                std::memcpy(values + i / sizeof(ElementType), &val, sizeof(int4));
            }
        }
        else if constexpr (bytes == sizeof(int2))
        {
            int2 val = *reinterpret_cast<const int2 *>(address);
            ;
            std::memcpy(values, &val, sizeof(int2));
        }
        else if constexpr (bytes == sizeof(int1))
        {
            int1 val = *reinterpret_cast<const int1 *>(address);
            std::memcpy(values, &val, sizeof(int1));
        }
        else
        {
            std::copy(address, address + size, values);
        }
    }

    ElementType values[size];
};