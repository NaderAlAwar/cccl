// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

template <typename T>
thrust::device_vector<T> make_bounded_vector(std::size_t elements, bool zero_data)
{
  if (zero_data)
  {
    return thrust::device_vector<T>(elements, T{});
  }

  // nvbench_helper::generate is not instantiated for half/bfloat, so generate
  // float data and convert to the benchmark value type.
  thrust::device_vector<float> source = generate(elements, bit_entropy::_1_000, -1.0f, 1.0f);

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    return source;
  }
  else
  {
    thrust::device_vector<T> destination(elements, thrust::no_init);
    thrust::transform(source.begin(), source.end(), destination.begin(), [] __device__(float x) {
      return static_cast<T>(x);
    });
    return destination;
  }
}

template <typename T>
void cub_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  constexpr float rms_norm_eps = 1e-5f;
  const int batch_size         = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size        = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data         = state.get_int64("ZeroData") != 0;

  const auto elements  = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
  const auto num_items = batch_size * hidden_size;

  thrust::device_vector<T> input = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> output(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size), zero_data);
  thrust::device_vector<float> rms_rcp(batch_size, thrust::no_init);

  auto* d_input   = thrust::raw_pointer_cast(input.data());
  auto* d_output  = thrust::raw_pointer_cast(output.data());
  auto* d_weight  = thrust::raw_pointer_cast(weight.data());
  auto* d_rms_rcp = thrust::raw_pointer_cast(rms_rcp.data());

  auto squared_input = cuda::make_transform_iterator(d_input, [] __device__(T x) {
    const float value = static_cast<float>(x);
    return value * value;
  });
  auto reciprocal_output =
    cuda::make_transform_output_iterator(d_rms_rcp, [hidden_size, rms_norm_eps] __device__(float sum_of_squares) {
      return rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + rms_norm_eps);
    });

  std::size_t temp_storage_bytes = 0;
  NVBENCH_CUDA_CALL(cub::DeviceSegmentedReduce::Sum(
    nullptr, temp_storage_bytes, squared_input, reciprocal_output, batch_size, hidden_size));

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  auto* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Input");
  state.add_global_memory_reads<T>(hidden_size, "Weight");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    NVBENCH_CUDA_CALL(cub::DeviceSegmentedReduce::Sum(
      d_temp_storage,
      temp_storage_bytes,
      squared_input,
      reciprocal_output,
      batch_size,
      hidden_size,
      launch.get_stream()));

    NVBENCH_CUDA_CALL(cub::DeviceTransform::Transform(
      cuda::std::tuple{cuda::counting_iterator<int>{0}, d_input},
      d_output,
      num_items,
      [hidden_size_div = cuda::fast_mod_div<int>{hidden_size}, d_rms_rcp, d_weight] __device__(int idx, T x) {
        const int row     = idx / hidden_size_div;
        const int col     = idx % hidden_size_div;
        const float scale = static_cast<float>(d_weight[col]) * d_rms_rcp[row];

        return static_cast<T>(static_cast<float>(x) * scale);
      },
      launch.get_stream()));
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

NVBENCH_BENCH_TYPES(cub_rmsnorm, NVBENCH_TYPE_AXES(value_types))
  .set_name("cub_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 800, 150000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize",
                  {512,  768,  896,  1024, 1152, 1280, 1536, 1600, 2048, 2304,  2560,  2880,  3072,  3584,  3840,
                   4096, 4608, 4868, 5120, 6144, 6656, 7168, 8192, 9736, 12288, 12980, 16384, 18432, 19472, 39572});
