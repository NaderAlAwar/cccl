// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"
#include <flashinfer/norm.cuh>

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
cudaError_t rms_norm_flashinfer(
  T* d_input,
  T* d_weight,
  T* d_output,
  cuda::std::uint32_t batch_size,
  cuda::std::uint32_t hidden_size,
  float eps,
  bool enable_pdl     = false,
  cudaStream_t stream = 0)
{
  return flashinfer::norm::RMSNorm(
    d_input, d_weight, d_output, batch_size, hidden_size, hidden_size, hidden_size, eps, enable_pdl, stream);
}

template <typename T>
void flashinfer_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data  = state.get_int64("ZeroData") != 0;
  const auto elements   = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
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

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    NVBENCH_CUDA_CALL(rms_norm_flashinfer(
      d_input,
      d_weight,
      d_output,
      static_cast<cuda::std::uint32_t>(batch_size),
      static_cast<cuda::std::uint32_t>(hidden_size),
      rms_norm_eps,
      false,
      launch.get_stream()));
  });

#if 0
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

NVBENCH_BENCH_TYPES(flashinfer_rmsnorm, NVBENCH_TYPE_AXES(value_types))
  .set_name("flashinfer_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 800, 150000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize",
                  {512,  768,  896,  1024, 1152, 1280, 1536, 1600, 2048, 2304,  2560,  2880,  3072,  3584,  3840,
                   4096, 4608, 4868, 5120, 6144, 6656, 7168, 8192, 9736, 12288, 12980, 16384, 18432, 19472, 39572});
