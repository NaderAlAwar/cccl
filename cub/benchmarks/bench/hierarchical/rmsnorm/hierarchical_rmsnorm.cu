// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_hierarchical_transform.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

template <typename T>
int align_hidden_size(const nvbench::state& state, int hidden_size)
{
  const auto& device              = state.get_device();
  const int sm_version            = device.has_value() ? device->get_sm_version() : 0;
  const int alignment_bytes       = (sm_version >= 900 && sm_version < 1000) ? 128 : 16;
  const std::size_t row_bytes     = static_cast<std::size_t>(hidden_size) * sizeof(T);
  const std::size_t aligned_bytes = ((row_bytes + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
  return static_cast<int>(aligned_bytes / sizeof(T));
}

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
void hierarchical_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  constexpr float rms_norm_eps    = 1e-5f;
  const int batch_size            = static_cast<int>(state.get_int64("BatchSize"));
  const int requested_hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const int hidden_size           = align_hidden_size<T>(state, requested_hidden_size);
  const bool zero_data            = state.get_int64("ZeroData") != 0;

  const auto elements = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);

  thrust::device_vector<T> input = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> output(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size), zero_data);

  auto* d_input  = thrust::raw_pointer_cast(input.data());
  auto* d_output = thrust::raw_pointer_cast(output.data());
  auto* d_weight = thrust::raw_pointer_cast(weight.data());

  // The weight vector is reused by every row. LOAD_LDG gives those loads the
  // read-only cache path, which measured better than ordinary global loads for
  // this benchmark.
  cub::CacheModifiedInputIterator<cub::LOAD_LDG, T> d_weight_ldg(d_weight);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Input");
  state.add_global_memory_reads<T>(hidden_size, "Weight");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    NVBENCH_CUDA_CALL(cub::DeviceSegmentedTransform::TransformProlog(
      d_input,
      d_weight_ldg, // Right now this is a special case because its size is hidden_size, not batch_size * hidden_size.
                    // Later, we will need an iterator that expresses repeating hidden_size elements for each batch.
      d_output,
      batch_size,
      hidden_size,
      // Later: if we had variable segment sizes, this hidden_size would not work.
      // Can we make range hierarchical? i.e., range would be an object that is
      // constructed by the kernel, passed in. Mental model is two iterators, begin
      // and end.
      [hidden_size, rms_norm_eps] __device__(auto group, const auto& range) -> float {
        float partial_sum = 0.0f;

        range.for_each([&](auto item) {
          const float value = static_cast<float>(item);
          partial_sum += value * value;
        });

        const float sum_of_squares = cuda::coop::sum(group, partial_sum);

        return rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + rms_norm_eps);
      },
      [] __device__(T x, T weight, float rms_rcp) {
        const float scale = static_cast<float>(weight) * rms_rcp;

        return static_cast<T>(static_cast<float>(x) * scale);
      },
      launch.get_stream()));
  });

  rmsnorm_check::check_correctness(batch_size, hidden_size, rms_norm_eps, input, weight, output);
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
