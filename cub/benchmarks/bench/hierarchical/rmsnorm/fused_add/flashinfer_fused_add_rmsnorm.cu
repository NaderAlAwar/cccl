// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

#include <flashinfer/norm.cuh>

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
cudaError_t fused_add_rms_norm_flashinfer(
  T* d_input,
  T* d_residual,
  T* d_weight,
  cuda::std::uint32_t batch_size,
  cuda::std::uint32_t hidden_size,
  float eps,
  bool enable_pdl     = false,
  cudaStream_t stream = 0)
{
  return flashinfer::norm::FusedAddRMSNorm(
    d_input, d_residual, d_weight, batch_size, hidden_size, hidden_size, hidden_size, eps, enable_pdl, stream);
}

template <typename T>
void flashinfer_fused_add_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  constexpr float rms_norm_eps = 1e-5f;
  const int batch_size         = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size        = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data         = state.get_int64("ZeroData") != 0;
  const auto elements          = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
  const auto bytes             = elements * sizeof(T);

  thrust::device_vector<T> input_master    = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> residual_master = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> input_working(elements, thrust::no_init);
  thrust::device_vector<T> residual_working(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size), zero_data);

  auto* d_input_master     = thrust::raw_pointer_cast(input_master.data());
  auto* d_residual_master  = thrust::raw_pointer_cast(residual_master.data());
  auto* d_input_working    = thrust::raw_pointer_cast(input_working.data());
  auto* d_residual_working = thrust::raw_pointer_cast(residual_working.data());
  auto* d_weight           = thrust::raw_pointer_cast(weight.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Input");
  state.add_global_memory_reads<T>(elements, "Residual");
  state.add_global_memory_reads<T>(elements, "Weight");
  state.add_global_memory_writes<T>(elements, "Residual");
  state.add_global_memory_writes<T>(elements, "Output");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    NVBENCH_CUDA_CALL(
      cudaMemcpyAsync(d_input_working, d_input_master, bytes, cudaMemcpyDeviceToDevice, launch.get_stream()));
    NVBENCH_CUDA_CALL(
      cudaMemcpyAsync(d_residual_working, d_residual_master, bytes, cudaMemcpyDeviceToDevice, launch.get_stream()));

    timer.start();
    NVBENCH_CUDA_CALL(fused_add_rms_norm_flashinfer(
      d_input_working,
      d_residual_working,
      d_weight,
      static_cast<cuda::std::uint32_t>(batch_size),
      static_cast<cuda::std::uint32_t>(hidden_size),
      rms_norm_eps,
      false,
      launch.get_stream()));
    timer.stop();
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

NVBENCH_BENCH_TYPES(flashinfer_fused_add_rmsnorm, NVBENCH_TYPE_AXES(value_types))
  .set_name("flashinfer_fused_add_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 8192, 20000, 75000, 150000, 299000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize", {2880, 7168});
