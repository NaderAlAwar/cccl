// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <thrust/transform.h>

#include <cuda/std/type_traits>

#include <stdexcept>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

template <typename T>
struct convert_op
{
  __device__ T operator()(float x) const
  {
    return static_cast<T>(x);
  }
};

template <typename T>
struct unary_proxy_op
{
  __device__ T operator()(T x) const
  {
    return x;
  }
};

template <typename T>
struct thread_weight_proxy_op
{
  const T* weight;
  int hidden_size;

  __device__ T operator()(T x) const
  {
    // Deliberately use the physical thread index, not a logical element index.
    // We want a cheap pressure probe for "unary transform plus one weight read"
    // without adding a counting iterator, divide/mod by the global item index,
    // or row-RMS state. This is not RMSNorm-correct: for large hidden sizes it
    // repeatedly samples only a small prefix of the weight vector.
    const int col     = static_cast<int>(threadIdx.x) % hidden_size;
    const float scale = static_cast<float>(weight[col]);

    return static_cast<T>(static_cast<float>(x) * scale);
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
void device_transform_rmsnorm_proxy(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data  = state.get_int64("ZeroData") != 0;
  const auto proxy_mode = state.get_string("ProxyMode");
  const auto elements   = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
  const auto num_items  = batch_size * hidden_size;

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
  state.add_global_memory_writes<T>(elements, "Output");

  // This file intentionally contains bandwidth probes, not correct RMSNorm
  // implementations. We keep the reported denominator equal to the logical
  // RMSNorm traffic (input + weight + output) so the plots are comparable, then
  // use two simple DeviceTransform shapes to bound how much of the gap is due to
  // transform mechanics rather than the row reduction.
  //
  // `unary` is the clean one-input/one-output DeviceTransform roof.
  // `thread_weight` keeps that shape but forces one cheap weight load. It avoids
  // a counting iterator because the goal is a pressure probe, not reconstructing
  // the semantic `(row, col)` position inside a flat transform.
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    if (proxy_mode == "unary")
    {
      NVBENCH_CUDA_CALL(
        cub::DeviceTransform::Transform(d_input, d_output, num_items, unary_proxy_op<T>{}, launch.get_stream()));
    }
    else if (proxy_mode == "thread_weight")
    {
      NVBENCH_CUDA_CALL(cub::DeviceTransform::Transform(
        d_input, d_output, num_items, thread_weight_proxy_op<T>{d_weight, hidden_size}, launch.get_stream()));
    }
  });
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

NVBENCH_BENCH_TYPES(device_transform_rmsnorm_proxy, NVBENCH_TYPE_AXES(value_types))
  .set_name("device_transform_rmsnorm_proxy")
  .set_type_axes_names({"T{ct}"})
  .add_string_axis("ProxyMode", {"unary", "thread_weight"})
  .add_int64_axis("BatchSize", {64, 800, 150000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize",
                  {512,  768,  896,  1024, 1152, 1280, 1536, 1600, 2048, 2304,  2560,  2880,  3072,  3584,  3840,
                   4096, 4608, 4868, 5120, 6144, 6656, 7168, 8192, 9736, 12288, 12980, 16384, 18432, 19472, 39572});
