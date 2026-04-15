/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/tabulate.h>

#include <memory>
#include <type_traits>
#include <vector>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cub::benchmarks::cudf_input
{
namespace detail
{
template <typename T>
struct random_value_generator
{
  unsigned seed;

  __device__ T operator()(cudf::size_type index) const
  {
    thrust::minstd_rand engine(seed);
    engine.discard(index);

    if constexpr (std::is_same_v<T, bool>)
    {
      thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
      return dist(engine) < 0.5f;
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
      thrust::uniform_real_distribution<T> dist(static_cast<T>(-1024), static_cast<T>(1024));
      return dist(engine);
    }
    else
    {
      thrust::uniform_int_distribution<T> dist(static_cast<T>(-1024), static_cast<T>(1024));
      return dist(engine);
    }
  }
};

struct validity_generator
{
  unsigned seed;
  double null_probability;

  __device__ bool operator()(cudf::size_type index) const
  {
    thrust::minstd_rand engine(seed);
    engine.discard(index);
    thrust::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(engine) >= null_probability;
  }
};
} // namespace detail

template <typename T>
std::unique_ptr<cudf::column> make_random_fixed_width_column(
  cudf::size_type size, unsigned value_seed, bool nullable, unsigned validity_seed, double null_probability = 0.01)
{
  auto stream = cudf::get_default_stream();
  auto column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<T>()}, size);

  auto mutable_view = column->mutable_view();
  thrust::tabulate(thrust::cuda::par.on(stream.value()),
                   mutable_view.begin<T>(),
                   mutable_view.end<T>(),
                   detail::random_value_generator<T>{value_seed});

  if (nullable)
  {
    auto begin                   = thrust::make_counting_iterator<cudf::size_type>(0);
    auto end                     = begin + size;
    auto [null_mask, null_count] = cudf::detail::valid_if(
      begin,
      end,
      detail::validity_generator{validity_seed, null_probability},
      stream,
      cudf::get_current_device_resource_ref());
    column->set_null_mask(std::move(null_mask), null_count);
  }

  return column;
}

template <typename T>
std::unique_ptr<cudf::table> make_copy_if_else_input(cudf::size_type num_rows, bool nullable)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(make_random_fixed_width_column<T>(num_rows, 17u, nullable, 29u));
  columns.emplace_back(make_random_fixed_width_column<T>(num_rows, 31u, nullable, 43u));
  columns.emplace_back(make_random_fixed_width_column<bool>(num_rows, 59u, nullable, 71u));
  return std::make_unique<cudf::table>(std::move(columns));
}
} // namespace cub::benchmarks::cudf_input
