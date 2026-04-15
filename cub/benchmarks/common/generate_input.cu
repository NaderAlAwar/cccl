/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <cuda/std/algorithm>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "generate_input.hpp"
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace
{
thrust::minstd_rand deterministic_engine(unsigned seed)
{
  return thrust::minstd_rand{seed};
}

template <typename T>
double std_dev_from_range(T lower_bound, T upper_bound)
{
  return std::abs(static_cast<double>(upper_bound) - static_cast<double>(lower_bound)) / 6.0;
}

template <typename T>
auto make_normal_dist(T lower_bound, T upper_bound)
{
  using real_type    = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  auto const mean    = static_cast<real_type>(lower_bound) / 2 + static_cast<real_type>(upper_bound) / 2;
  auto const std_dev = static_cast<real_type>(std_dev_from_range(lower_bound, upper_bound));
  return thrust::random::normal_distribution<real_type>(mean, std_dev);
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
auto make_uniform_dist(T lower_bound, T upper_bound)
{
  return thrust::uniform_int_distribution<T>(lower_bound, upper_bound);
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
auto make_uniform_dist(T lower_bound, T upper_bound)
{
  return thrust::uniform_real_distribution<T>(lower_bound, upper_bound);
}

template <typename T>
class geometric_distribution : public thrust::random::normal_distribution<double>
{
public:
  geometric_distribution(T lower_bound, T upper_bound)
      : thrust::random::normal_distribution<double>(0.0, std_dev_from_range(lower_bound, upper_bound) * 2.0)
      , lower_bound_(lower_bound)
      , upper_bound_(upper_bound)
  {}

  template <typename Generator>
  __host__ __device__ T operator()(Generator& generator)
  {
    auto const base = std::abs(thrust::random::normal_distribution<double>::operator()(generator));
    auto const value =
      lower_bound_ < upper_bound_ ? base + static_cast<double>(lower_bound_) : static_cast<double>(lower_bound_) - base;
    if constexpr (std::is_integral_v<T>)
    {
      return static_cast<T>(::round(value));
    }
    else
    {
      return static_cast<T>(value);
    }
  }

private:
  T lower_bound_;
  T upper_bound_;
};

struct bool_generator
{
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> dist{0.0f, 1.0f};
  double probability_true;

  bool_generator(thrust::minstd_rand engine, double probability_true)
      : engine(engine)
      , probability_true(probability_true)
  {}

  __device__ bool operator()(std::size_t idx)
  {
    engine.discard(idx);
    return dist(engine) < probability_true;
  }
};

template <typename T, typename Distribution>
struct value_generator
{
  T lower_bound;
  T upper_bound;
  thrust::minstd_rand engine;
  Distribution dist;

  __device__ T operator()(std::size_t idx)
  {
    engine.discard(idx);
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>)
    {
      auto const value   = static_cast<double>(dist(engine));
      auto const clamped = cuda::std::clamp(value, static_cast<double>(lower_bound), static_cast<double>(upper_bound));
      return static_cast<T>(::round(clamped));
    }
    else
    {
      return cuda::std::clamp(dist(engine), lower_bound, upper_bound);
    }
  }
};

std::pair<rmm::device_buffer, cudf::size_type>
create_random_null_mask(cudf::size_type num_rows, std::optional<double> p, unsigned seed)
{
  if (!p.has_value())
  {
    return {rmm::device_buffer{}, 0};
  }

  auto const probability_valid = 1.0 - *p;
  return cudf::detail::valid_if(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    [generator = bool_generator{deterministic_engine(seed), probability_valid}] __device__(cudf::size_type idx) mutable {
      return generator(idx);
    },
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
}

template <typename T>
std::unique_ptr<cudf::column>
make_random_numeric_column(cudf::size_type num_rows, data_profile const& profile, unsigned seed)
{
  auto const params = profile.get_distribution_params<T>();
  auto const stream = cudf::get_default_stream();
  auto mutable_col  = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()},
    num_rows,
    cudf::mask_state::UNALLOCATED,
    stream,
    cudf::get_current_device_resource_ref());
  auto view         = mutable_col->mutable_view();
  auto output_begin = static_cast<T*>(view.head<T>());
  auto engine       = deterministic_engine(seed);

  switch (params.id)
  {
    case distribution_id::NORMAL: {
      auto dist = make_normal_dist<T>(params.lower_bound, params.upper_bound);
      thrust::tabulate(thrust::device,
                       output_begin,
                       output_begin + num_rows,
                       value_generator<T, decltype(dist)>{params.lower_bound, params.upper_bound, engine, dist});
      break;
    }
    case distribution_id::UNIFORM: {
      auto dist = make_uniform_dist<T>(params.lower_bound, params.upper_bound);
      thrust::tabulate(thrust::device,
                       output_begin,
                       output_begin + num_rows,
                       value_generator<T, decltype(dist)>{params.lower_bound, params.upper_bound, engine, dist});
      break;
    }
    case distribution_id::GEOMETRIC: {
      auto dist = geometric_distribution<T>(params.lower_bound, params.upper_bound);
      thrust::tabulate(thrust::device,
                       output_begin,
                       output_begin + num_rows,
                       value_generator<T, decltype(dist)>{params.lower_bound, params.upper_bound, engine, dist});
      break;
    }
  }

  auto [mask, null_count] = create_random_null_mask(num_rows, profile.get_null_probability(), seed + 1);
  mutable_col->set_null_mask(std::move(mask), null_count);
  return mutable_col;
}

template <typename T>
std::unique_ptr<cudf::column>
make_random_chrono_column(cudf::size_type num_rows, data_profile const& profile, unsigned seed)
{
  auto const stream = cudf::get_default_stream();
  auto mutable_col  = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()},
    num_rows,
    cudf::mask_state::UNALLOCATED,
    stream,
    cudf::get_current_device_resource_ref());
  auto view         = mutable_col->mutable_view();
  auto output_begin = static_cast<T*>(view.head<T>());

  auto const rep_params = profile.get_distribution_params<typename T::rep>();
  auto engine           = deterministic_engine(seed);

  switch (rep_params.id)
  {
    case distribution_id::NORMAL: {
      auto dist = make_normal_dist<typename T::rep>(rep_params.lower_bound, rep_params.upper_bound);
      thrust::tabulate(
        thrust::device,
        output_begin,
        output_begin + num_rows,
        [generator = value_generator<typename T::rep, decltype(dist)>{
           rep_params.lower_bound, rep_params.upper_bound, engine, dist}] __device__(std::size_t idx) mutable {
          return T{generator(idx)};
        });
      break;
    }
    case distribution_id::UNIFORM: {
      auto dist = make_uniform_dist<typename T::rep>(rep_params.lower_bound, rep_params.upper_bound);
      thrust::tabulate(
        thrust::device,
        output_begin,
        output_begin + num_rows,
        [generator = value_generator<typename T::rep, decltype(dist)>{
           rep_params.lower_bound, rep_params.upper_bound, engine, dist}] __device__(std::size_t idx) mutable {
          return T{generator(idx)};
        });
      break;
    }
    case distribution_id::GEOMETRIC: {
      auto dist = geometric_distribution<typename T::rep>(rep_params.lower_bound, rep_params.upper_bound);
      thrust::tabulate(
        thrust::device,
        output_begin,
        output_begin + num_rows,
        [generator = value_generator<typename T::rep, decltype(dist)>{
           rep_params.lower_bound, rep_params.upper_bound, engine, dist}] __device__(std::size_t idx) mutable {
          return T{generator(idx)};
        });
      break;
    }
  }

  auto [mask, null_count] = create_random_null_mask(num_rows, profile.get_null_probability(), seed + 1);
  mutable_col->set_null_mask(std::move(mask), null_count);
  return mutable_col;
}

std::unique_ptr<cudf::column>
make_random_bool_column(cudf::size_type num_rows, data_profile const& profile, unsigned seed)
{
  auto const params = profile.get_distribution_params<bool>();
  auto const stream = cudf::get_default_stream();
  auto mutable_col  = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8},
    num_rows,
    cudf::mask_state::UNALLOCATED,
    stream,
    cudf::get_current_device_resource_ref());
  auto view         = mutable_col->mutable_view();
  auto output_begin = static_cast<bool*>(view.head<bool>());

  thrust::tabulate(thrust::device,
                   output_begin,
                   output_begin + num_rows,
                   bool_generator{deterministic_engine(seed), params.probability_true});

  auto [mask, null_count] = create_random_null_mask(num_rows, profile.get_null_probability(), seed + 1);
  mutable_col->set_null_mask(std::move(mask), null_count);
  return mutable_col;
}

template <typename T>
std::unique_ptr<cudf::column>
make_sequence_column(cudf::size_type num_rows, std::optional<double> null_probability, unsigned seed)
{
  auto init = cudf::make_default_constructed_scalar(cudf::data_type{cudf::type_to_id<T>()});
  init->set_valid_async(true, cudf::get_default_stream());
  auto col                = cudf::sequence(num_rows, *init);
  auto [mask, null_count] = create_random_null_mask(num_rows, null_probability, seed);
  col->set_null_mask(std::move(mask), null_count);
  return col;
}

std::unique_ptr<cudf::column> create_supported_random_column(
  cudf::type_id dtype_id, cudf::size_type num_rows, data_profile const& profile, unsigned seed)
{
  switch (dtype_id)
  {
    case cudf::type_id::INT8:
      return make_random_numeric_column<std::int8_t>(num_rows, profile, seed);
    case cudf::type_id::INT16:
      return make_random_numeric_column<std::int16_t>(num_rows, profile, seed);
    case cudf::type_id::INT32:
      return make_random_numeric_column<std::int32_t>(num_rows, profile, seed);
    case cudf::type_id::INT64:
      return make_random_numeric_column<std::int64_t>(num_rows, profile, seed);
    case cudf::type_id::UINT8:
      return make_random_numeric_column<std::uint8_t>(num_rows, profile, seed);
    case cudf::type_id::UINT16:
      return make_random_numeric_column<std::uint16_t>(num_rows, profile, seed);
    case cudf::type_id::UINT32:
      return make_random_numeric_column<std::uint32_t>(num_rows, profile, seed);
    case cudf::type_id::UINT64:
      return make_random_numeric_column<std::uint64_t>(num_rows, profile, seed);
    case cudf::type_id::FLOAT32:
      return make_random_numeric_column<float>(num_rows, profile, seed);
    case cudf::type_id::FLOAT64:
      return make_random_numeric_column<double>(num_rows, profile, seed);
    case cudf::type_id::BOOL8:
      return make_random_bool_column(num_rows, profile, seed);
    default:
      CUDF_FAIL("CCCL benchmark input generation only supports fixed-width numeric and bool types.");
  }
}

std::unique_ptr<cudf::column> create_supported_sequence_column(
  cudf::type_id dtype_id, cudf::size_type num_rows, std::optional<double> null_probability, unsigned seed)
{
  switch (dtype_id)
  {
    case cudf::type_id::INT8:
      return make_sequence_column<std::int8_t>(num_rows, null_probability, seed);
    case cudf::type_id::INT16:
      return make_sequence_column<std::int16_t>(num_rows, null_probability, seed);
    case cudf::type_id::INT32:
      return make_sequence_column<std::int32_t>(num_rows, null_probability, seed);
    case cudf::type_id::INT64:
      return make_sequence_column<std::int64_t>(num_rows, null_probability, seed);
    case cudf::type_id::UINT8:
      return make_sequence_column<std::uint8_t>(num_rows, null_probability, seed);
    case cudf::type_id::UINT16:
      return make_sequence_column<std::uint16_t>(num_rows, null_probability, seed);
    case cudf::type_id::UINT32:
      return make_sequence_column<std::uint32_t>(num_rows, null_probability, seed);
    case cudf::type_id::UINT64:
      return make_sequence_column<std::uint64_t>(num_rows, null_probability, seed);
    case cudf::type_id::FLOAT32:
      return make_sequence_column<float>(num_rows, null_probability, seed);
    case cudf::type_id::FLOAT64:
      return make_sequence_column<double>(num_rows, null_probability, seed);
    case cudf::type_id::BOOL8:
      return make_sequence_column<bool>(num_rows, null_probability, seed);
    default:
      CUDF_FAIL("CCCL benchmark sequence generation only supports fixed-width numeric and bool types.");
  }
}
} // namespace

std::unique_ptr<cudf::table> create_random_table(
  std::vector<cudf::type_id> const& dtype_ids, row_count num_rows, data_profile const& profile, unsigned seed)
{
  auto seed_engine = deterministic_engine(seed);
  thrust::uniform_int_distribution<unsigned> seed_dist;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(dtype_ids.size());
  for (auto const dtype_id : dtype_ids)
  {
    columns.push_back(create_supported_random_column(dtype_id, num_rows.count, profile, seed_dist(seed_engine)));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> create_random_table(
  std::vector<cudf::type_id> const& dtype_ids, std::size_t num_rows, data_profile const& profile, unsigned seed)
{
  return create_random_table(dtype_ids, row_count{static_cast<cudf::size_type>(num_rows)}, profile, seed);
}

std::unique_ptr<cudf::table> create_sequence_table(
  std::vector<cudf::type_id> const& dtype_ids, row_count num_rows, std::optional<double> null_probability, unsigned seed)
{
  auto seed_engine = deterministic_engine(seed);
  thrust::uniform_int_distribution<unsigned> seed_dist;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(dtype_ids.size());
  for (auto const dtype_id : dtype_ids)
  {
    columns.push_back(
      create_supported_sequence_column(dtype_id, num_rows.count, null_probability, seed_dist(seed_engine)));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}
