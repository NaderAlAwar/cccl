// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

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
struct add_input_and_residual_op
{
  __device__ T operator()(T input, T residual) const
  {
    return static_cast<T>(static_cast<float>(input) + static_cast<float>(residual));
  }
};

template <typename T>
struct square_op
{
  __device__ float operator()(T x) const
  {
    const float value = static_cast<float>(x);
    return value * value;
  }
};

struct reciprocal_rms_op
{
  int hidden_size;
  float eps;

  __device__ float operator()(float sum_of_squares) const
  {
    return rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + eps);
  }
};

template <typename T>
struct normalize_and_scale_op
{
  cuda::fast_mod_div<int> hidden_size;
  const float* rms_rcp;
  const T* weight;

  __device__ T operator()(int idx, T x) const
  {
    const int row     = idx / hidden_size;
    const int col     = idx % hidden_size;
    const float scale = static_cast<float>(weight[col]) * rms_rcp[row];

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
void fused_add_rmsnorm(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data  = state.get_int64("ZeroData") != 0;
  const auto elements   = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);
  const auto num_items  = batch_size * hidden_size;
  const auto bytes      = elements * sizeof(T);

  thrust::device_vector<T> input_master    = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> residual_master = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> input_working(elements, thrust::no_init);
  thrust::device_vector<T> residual_working(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size), zero_data);
  thrust::device_vector<float> rms_rcp(batch_size, thrust::no_init);

  auto* d_input_master     = thrust::raw_pointer_cast(input_master.data());
  auto* d_residual_master  = thrust::raw_pointer_cast(residual_master.data());
  auto* d_input_working    = thrust::raw_pointer_cast(input_working.data());
  auto* d_residual_working = thrust::raw_pointer_cast(residual_working.data());
  auto* d_weight           = thrust::raw_pointer_cast(weight.data());
  auto* d_rms_rcp          = thrust::raw_pointer_cast(rms_rcp.data());

  auto squared_residual = cuda::make_transform_iterator(d_residual_working, square_op<T>{});
  auto reciprocal_output =
    cuda::make_transform_output_iterator(d_rms_rcp, reciprocal_rms_op{hidden_size, rms_norm_eps});

  std::size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(
    nullptr, temp_storage_bytes, squared_residual, reciprocal_output, batch_size, hidden_size);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  auto* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Input");
  state.add_global_memory_reads<T>(3 * elements, "Residual");
  state.add_global_memory_reads<T>(elements, "Weight");
  state.add_global_memory_reads<float>(elements, "InvRMS");
  state.add_global_memory_writes<T>(elements, "Residual");
  state.add_global_memory_writes<float>(batch_size);
  state.add_global_memory_writes<T>(elements, "Output");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    NVBENCH_CUDA_CALL(
      cudaMemcpyAsync(d_input_working, d_input_master, bytes, cudaMemcpyDeviceToDevice, launch.get_stream()));
    NVBENCH_CUDA_CALL(
      cudaMemcpyAsync(d_residual_working, d_residual_master, bytes, cudaMemcpyDeviceToDevice, launch.get_stream()));

    timer.start();

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(d_input_working, d_residual_working),
      d_residual_working,
      num_items,
      add_input_and_residual_op<T>{},
      launch.get_stream());

    cub::DeviceSegmentedReduce::Sum(
      d_temp_storage,
      temp_storage_bytes,
      squared_residual,
      reciprocal_output,
      batch_size,
      hidden_size,
      launch.get_stream());

    cub::DeviceTransform::Transform(
      cuda::std::tuple{cuda::counting_iterator<int>{0}, d_residual_working},
      d_input_working,
      num_items,
      normalize_and_scale_op<T>{cuda::fast_mod_div<int>{hidden_size}, d_rms_rcp, d_weight},
      launch.get_stream());

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

NVBENCH_BENCH_TYPES(fused_add_rmsnorm, NVBENCH_TYPE_AXES(value_types))
  .set_name("fused_add_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 8192, 20000, 75000, 150000, 299000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize", {2880, 7168});
