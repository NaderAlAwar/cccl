// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rmsnorm_check
{
template <typename T>
constexpr float correctness_tolerance()
{
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    return 1e-4f;
  }
  else if constexpr (cuda::std::is_same_v<T, __half>)
  {
    return 5e-2f;
  }
  else
  {
    return 8e-2f;
  }
}

template <typename T>
void check_correctness(
  int batch_size,
  int hidden_size,
  float eps,
  const thrust::device_vector<T>& input,
  const thrust::device_vector<T>& weight,
  const thrust::device_vector<T>& output)
{
  const thrust::host_vector<T> h_input  = input;
  const thrust::host_vector<T> h_weight = weight;
  const thrust::host_vector<T> h_output = output;

  const float tolerance = correctness_tolerance<T>();

  for (int row = 0; row < batch_size; ++row)
  {
    const int row_offset = row * hidden_size;
    float sum_of_squares = 0.0f;

    for (int col = 0; col < hidden_size; ++col)
    {
      const float x = static_cast<float>(h_input[row_offset + col]);
      sum_of_squares += x * x;
    }

    const float rms_rcp = 1.0f / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) + eps);

    for (int col = 0; col < hidden_size; ++col)
    {
      const int idx        = row_offset + col;
      const float x        = static_cast<float>(h_input[idx]);
      const float w        = static_cast<float>(h_weight[col]);
      const float actual   = static_cast<float>(h_output[idx]);
      const float expected = x * rms_rcp * w;
      const float abs_diff = std::abs(actual - expected);
      const float rel_diff = abs_diff / std::max(1.0f, std::abs(expected));

      if (!std::isfinite(actual) || (abs_diff > tolerance && rel_diff > tolerance))
      {
        throw std::runtime_error("RMSNorm correctness check failed.");
      }
    }
  }
}
} // namespace rmsnorm_check
