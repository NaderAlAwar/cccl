// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

constexpr float rms_norm_eps = 1e-5f;

template <typename T>
struct convert_op
{
  __device__ T operator()(float x) const
  {
    return static_cast<T>(x);
  }
};

template <typename T>
thrust::device_vector<T> make_bounded_vector(std::size_t elements, bool zero_data)
{
  if (zero_data)
  {
    return thrust::device_vector<T>(elements, T{});
  }

  thrust::device_vector<float> source = generate(elements, bit_entropy::_1_000, -1.0f, 1.0f);

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    return source;
  }
  else
  {
    thrust::device_vector<T> destination(elements, thrust::no_init);
    thrust::transform(source.begin(), source.end(), destination.begin(), convert_op<T>{});
    return destination;
  }
}

template <typename T>
struct bad_rmsnorm_op
{
  cuda::fast_mod_div<int> hidden_size;
  int hidden_size_value;
  const T* input;
  const T* weight;
  float inv_hidden_size;
  float eps;

  __device__ T operator()(int idx, T x) const
  {
    const int col           = idx % hidden_size;
    const int segment_begin = idx - col;

    float sum_of_squares = 0.0f;
    for (int i = 0; i < hidden_size_value; ++i)
    {
      const float value = static_cast<float>(input[segment_begin + i]);
      sum_of_squares += value * value;
    }

    const float rms_rcp = rsqrtf(sum_of_squares * inv_hidden_size + eps);
    const float scale   = static_cast<float>(weight[col]) * rms_rcp;

    return static_cast<T>(static_cast<float>(x) * scale);
  }
};

template <typename T>
void bad_rmsnorm_transform_cub(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data  = state.get_int64("ZeroData") != 0;

  const auto elements  = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
  const auto num_items = batch_size * hidden_size;

  thrust::device_vector<T> input = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> output(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size), zero_data);

  auto* d_input  = thrust::raw_pointer_cast(input.data());
  auto* d_output = thrust::raw_pointer_cast(output.data());
  auto* d_weight = thrust::raw_pointer_cast(weight.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Input");
  state.add_global_memory_reads<T>(hidden_size, "Weight");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceTransform::Transform(
      cuda::std::tuple{cuda::counting_iterator<int>{0}, d_input},
      d_output,
      num_items,
      bad_rmsnorm_op<T>{
        cuda::fast_mod_div<int>{hidden_size},
        hidden_size,
        d_input,
        d_weight,
        1.0f / static_cast<float>(hidden_size),
        rms_norm_eps},
      launch.get_stream());
  });

#if 1
  rmsnorm_check::check_correctness(batch_size, hidden_size, rms_norm_eps, input, weight, output);
#endif
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types =
  nvbench::type_list<float
#  if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
                     ,
                     __half
#  endif
#  if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
                     ,
                     __nv_bfloat16
#  endif
                     >;
#endif

NVBENCH_BENCH_TYPES(bad_rmsnorm_transform_cub, NVBENCH_TYPE_AXES(value_types))
  .set_name("cub_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 8192, 20000, 75000, 150000, 299000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize", {2880, 7168});
