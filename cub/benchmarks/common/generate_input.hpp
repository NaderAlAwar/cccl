/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/type_traits>

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

enum class distribution_id : int8_t
{
  UNIFORM,
  NORMAL,
  GEOMETRIC,
};

template <typename T, typename Enable = void>
struct distribution_params;

template <typename T>
struct distribution_params<T, std::enable_if_t<!std::is_same_v<T, bool> && cudf::is_numeric<T>()>>
{
  distribution_id id;
  T lower_bound;
  T upper_bound;
};

template <typename T>
struct distribution_params<T, std::enable_if_t<std::is_same_v<T, bool>>>
{
  double probability_true;
};

class data_profile
{
public:
  template <typename T, std::enable_if_t<!std::is_same_v<T, bool> && cuda::std::is_integral_v<T>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const;

  template <typename T, std::enable_if_t<std::is_same_v<T, bool>, T>* = nullptr>
  distribution_params<T> get_distribution_params() const
  {
    return distribution_params<T>{bool_probability_true_};
  }

  std::optional<double> get_null_probability() const
  {
    return null_probability_;
  }

  template <typename T, std::enable_if_t<!std::is_same_v<T, bool> && cuda::std::is_integral_v<T>, T>* = nullptr>
  void set_distribution(distribution_params<T> params)
  {
    int_params_[cudf::type_to_id<T>()] =
      distribution_params<std::int64_t>{params.id, params.lower_bound, params.upper_bound};
  }

  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
  void set_distribution(distribution_params<T> params)
  {
    float_params_[cudf::type_to_id<T>()] = params;
  }

  void set_bool_probability_true(double p)
  {
    bool_probability_true_ = p;
  }
  void set_null_probability(std::optional<double> p)
  {
    null_probability_ = p;
  }

private:
  std::map<cudf::type_id, distribution_params<std::int64_t>> int_params_{};
  std::map<cudf::type_id, distribution_params<double>> float_params_{};
  double bool_probability_true_{0.5};
  std::optional<double> null_probability_{0.01};
};

class data_profile_builder
{
public:
  template <typename T>
  data_profile_builder& distribution(distribution_id id, T lower_bound, T upper_bound)
  {
    profile_.set_distribution<T>(distribution_params<T>{id, lower_bound, upper_bound});
    return *this;
  }

  data_profile_builder& bool_probability_true(double p)
  {
    profile_.set_bool_probability_true(p);
    return *this;
  }

  data_profile_builder& null_probability(std::optional<double> p)
  {
    profile_.set_null_probability(p);
    return *this;
  }

  data_profile_builder& no_validity()
  {
    profile_.set_null_probability(std::nullopt);
    return *this;
  }

  operator data_profile&&()
  {
    return std::move(profile_);
  }

private:
  data_profile profile_{};
};

struct row_count
{
  cudf::size_type count;
};

std::unique_ptr<cudf::table> create_random_table(
  std::vector<cudf::type_id> const& dtype_ids,
  row_count num_rows,
  data_profile const& data_params = data_profile{},
  unsigned seed                   = 0);

std::unique_ptr<cudf::table> create_random_table(
  std::vector<cudf::type_id> const& dtype_ids,
  std::size_t num_rows,
  data_profile const& data_params = data_profile{},
  unsigned seed                   = 0);

std::unique_ptr<cudf::table> create_sequence_table(
  std::vector<cudf::type_id> const& dtype_ids,
  row_count num_rows,
  std::optional<double> null_probability = std::nullopt,
  unsigned seed                          = 0);

template <typename T, std::enable_if_t<!std::is_same_v<T, bool> && cuda::std::is_integral_v<T>, T>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  auto const it = int_params_.find(cudf::type_to_id<T>());
  if (it == int_params_.end())
  {
    return distribution_params<T>{
      std::is_signed_v<T> ? distribution_id::NORMAL : distribution_id::GEOMETRIC,
      static_cast<T>(std::numeric_limits<T>::lowest() / 2),
      static_cast<T>(std::numeric_limits<T>::max() / 2)};
  }
  auto const& params = it->second;
  return distribution_params<T>{params.id, static_cast<T>(params.lower_bound), static_cast<T>(params.upper_bound)};
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, T>*>
distribution_params<T> data_profile::get_distribution_params() const
{
  auto const it = float_params_.find(cudf::type_to_id<T>());
  if (it == float_params_.end())
  {
    return distribution_params<T>{
      distribution_id::NORMAL, std::numeric_limits<T>::lowest() / 2, std::numeric_limits<T>::max() / 2};
  }
  auto const& params = it->second;
  return distribution_params<T>{params.id, static_cast<T>(params.lower_bound), static_cast<T>(params.upper_bound)};
}
