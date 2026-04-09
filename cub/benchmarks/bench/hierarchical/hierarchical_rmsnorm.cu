// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_hierarchical_transform.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

constexpr float rms_norm_eps             = 1e-5f;
constexpr int hierarchical_block_threads = 256;

template <typename T>
struct convert_op
{
  __device__ T operator()(float x) const
  {
    return static_cast<T>(x);
  }
};

template <typename T>
thrust::device_vector<T> make_bounded_vector(std::size_t elements)
{
  thrust::device_vector<float> source = generate(elements, bit_entropy::_1_000, -1.0f, 1.0f);

  if constexpr (::cuda::std::is_same_v<T, float>)
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

template <typename T, int BlockThreads>
struct rmsnorm_segment_op
{
  using result_type = float;

  int hidden_size;
  float eps;

  template <typename Group, typename Range>
  __device__ float operator()(Group group, const Range& range) const
  {
    float partial_sum = 0.0f;

    for (int item = 0; item < range.size(); ++item)
    {
      const float value = static_cast<float>(range[item]);
      partial_sum += value * value;
    }

    using block_reduce_t = cub::BlockReduce<float, BlockThreads>;
    __shared__ typename block_reduce_t::TempStorage temp_storage;

    const float sum_of_squares = block_reduce_t(temp_storage).Sum(partial_sum);

    if (::cuda::gpu_thread.is_root_rank(group))
    {
      return rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + eps);
    }

    return 0.0f;
  }
};

template <typename T>
struct hierarchical_normalize_and_scale_op
{
  const T* weight;

  __device__ T operator()(float rms_rcp, int index_in_segment, T x) const
  {
    const float scale = static_cast<float>(weight[index_in_segment]) * rms_rcp;

    return static_cast<T>(static_cast<float>(x) * scale);
  }
};

template <typename T>
void hierarchical_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));

  const auto elements = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);

  thrust::device_vector<T> input = make_bounded_vector<T>(elements);
  thrust::device_vector<T> output(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size));

  auto* d_input  = thrust::raw_pointer_cast(input.data());
  auto* d_output = thrust::raw_pointer_cast(output.data());
  auto* d_weight = thrust::raw_pointer_cast(weight.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements, "Input");
  state.add_global_memory_reads<T>(elements, "Weight");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceHierarchicalTransform::Transform(
      d_input,
      d_output,
      batch_size,
      hidden_size,
      rmsnorm_segment_op<T, hierarchical_block_threads>{hidden_size, rms_norm_eps},
      hierarchical_normalize_and_scale_op<T>{d_weight},
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
  .add_int64_axis("BatchSize", {64, 8192, 20000, 75000, 150000, 299000})
  .add_int64_axis("HiddenSize", {2880, 7168});
