// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_hierarchical_transform.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

constexpr float rms_norm_eps = 1e-5f;

struct rmsnorm_weight_source_offset_op
{
  int segment_size{};

  template <cuda::std::size_t SourceIndex>
  __host__ __device__ cuda::std::size_t operator()(
    cuda::std::integral_constant<cuda::std::size_t, SourceIndex>, cuda::std::size_t absolute_logical_offset, int) const
  {
    return absolute_logical_offset % static_cast<cuda::std::size_t>(segment_size);
  }
};

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
void hierarchical_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data  = state.get_int64("ZeroData") != 0;

  const auto elements = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
  if (rmsnorm_check::should_skip_large_tensor_on_affected_arch(state, elements))
  {
    state.skip("Skipping: large RMSNorm tensors above 2^31 elements on sm_90/sm_100.");
    return;
  }

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

  auto segment_op = [hidden_size, eps = rms_norm_eps] __device__(auto group, const auto& range) -> float {
    float partial_sum = 0.0f;

    // with a proper range can be range based for loop
    for (int item = 0; item < range.size(); ++item)
    {
      const float value = static_cast<float>(range[item]);
      partial_sum += value * value;
    }

    const float sum_of_squares = cuda::device::reduce(group, partial_sum, cuda::std::plus<>{});

    return rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + eps);
  };
  auto element_transform_op = [] __device__(float rms_rcp, int, auto input_item) {
    const float x     = static_cast<float>(cuda::std::get<0>(input_item));
    const float scale = static_cast<float>(cuda::std::get<1>(input_item)) * rms_rcp;

    return static_cast<T>(x * scale);
  };
  auto weight_logical   = cuda::constant_iterator<T>{T{}};
  auto weight_stageable = cub::detail::hierarchical::make_transform_prolog_stageable_iterator(
    weight_logical,
    cuda::std::make_tuple(d_weight),
    cub::detail::hierarchical::transform_prolog_first_staged_source{},
    rmsnorm_weight_source_offset_op{hidden_size});
  auto transform_input = cuda::std::make_tuple(d_input, weight_stageable);

  const cudaError_t warmup_error = cub::DeviceSegmentedTransform::TransformProlog(
    transform_input, d_output, batch_size, hidden_size, segment_op, element_transform_op);
  if (warmup_error == cudaErrorInvalidValue)
  {
    state.skip("Skipping: segment does not fit in dynamic shared memory.");
    return;
  }
  cudaDeviceSynchronize();

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceSegmentedTransform::TransformProlog(
      transform_input,
      d_output,
      batch_size,
      hidden_size,
      // Later: if we had variable segment sizes, this hidden_size would not work.
      // Can we make range hierarchical? i.e., range would be an object that is
      // constructed by the kernel, passed in. Mental model is two iterators, begin
      // and end.
      segment_op,
      element_transform_op,
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

NVBENCH_BENCH_TYPES(hierarchical_rmsnorm, NVBENCH_TYPE_AXES(value_types))
  .set_name("hierarchical_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 800, 150000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize",
                  {512,  768,  896,  1024, 1152, 1280, 1536, 1600, 2048, 2304,  2560,  2880,  3072,  3584,  3840,
                   4096, 4608, 4868, 5120, 6144, 6656, 7168, 8192, 9736, 12288, 12980, 16384, 18432, 19472, 39572});
