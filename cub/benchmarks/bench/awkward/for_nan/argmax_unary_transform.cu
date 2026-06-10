// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>

#include <cuda/iterator>
#include <cuda/std/limits>

#include <cstdint>
#include <random>
#include <vector>

#include <nvbench_helper.cuh>

template <typename T>
struct argmax_pair
{
  T value;
  std::int64_t local_index;
};

template <typename T>
struct argmax_pair_op
{
  __host__ __device__ argmax_pair<T> operator()(argmax_pair<T> lhs, argmax_pair<T> rhs) const
  {
    if (rhs.value > lhs.value || (rhs.value == lhs.value && rhs.local_index < lhs.local_index))
    {
      return rhs;
    }
    return lhs;
  }
};

template <typename T, typename OffsetT>
struct make_argmax_pair_op
{
  const T* values;
  const OffsetT* local_indices;

  __host__ __device__ argmax_pair<T> operator()(OffsetT idx) const
  {
    return {values[idx], static_cast<std::int64_t>(local_indices[idx])};
  }
};

template <typename OffsetT>
struct small_segments
{
  thrust::device_vector<OffsetT> offsets;
  thrust::device_vector<OffsetT> local_indices;
  std::size_t elements{};
};

template <typename OffsetT>
static small_segments<OffsetT> make_random_0_to_5_segments(std::size_t segments)
{
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> length_dist(0, 5);

  std::vector<OffsetT> h_offsets(segments + 1);
  std::vector<OffsetT> h_local_indices;
  h_offsets[0] = 0;

  OffsetT total = 0;
  for (std::size_t segment = 0; segment < segments; ++segment)
  {
    const int length = length_dist(rng);
    for (int local = 0; local < length; ++local)
    {
      h_local_indices.push_back(static_cast<OffsetT>(local));
    }
    total += static_cast<OffsetT>(length);
    h_offsets[segment + 1] = total;
  }

  return {thrust::device_vector<OffsetT>(h_offsets.begin(), h_offsets.end()),
          thrust::device_vector<OffsetT>(h_local_indices.begin(), h_local_indices.end()),
          static_cast<std::size_t>(total)};
}

template <typename T>
static void cccl_argmax_new_transform(nvbench::state& state, nvbench::type_list<T>)
{
  using offset_t = std::int64_t;

  const auto segments = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  auto segment_data   = make_random_0_to_5_segments<offset_t>(segments);

  thrust::device_vector<T> input = generate(segment_data.elements, bit_entropy::_1_000, T{0}, T{1});
  thrust::device_vector<std::int64_t> output(segments);

  const auto* values      = thrust::raw_pointer_cast(input.data());
  const auto* offsets_ptr = thrust::raw_pointer_cast(segment_data.offsets.data());
  auto* output_ptr        = thrust::raw_pointer_cast(output.data());

  auto op = [=] __device__(offset_t segment_id) -> std::int64_t {
    const offset_t begin = offsets_ptr[segment_id];
    const offset_t end   = offsets_ptr[segment_id + 1];

    T best_value               = cuda::std::numeric_limits<T>::lowest();
    std::int64_t best_position = -1;
    for (offset_t idx = begin; idx < end; ++idx)
    {
      const auto local = static_cast<std::int64_t>(idx - begin);
      const T value    = values[idx];
      if (best_position == -1 || value > best_value || (value == best_value && local < best_position))
      {
        best_value    = value;
        best_position = local;
      }
    }

    return best_position;
  };

  state.add_element_count(segment_data.elements);
  state.add_global_memory_reads<T>(segment_data.elements, "Size");
  state.add_global_memory_reads<offset_t>(segment_data.offsets.size());
  state.add_global_memory_writes<std::int64_t>(segments);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceTransform::Transform(
      cuda::counting_iterator<offset_t>{0}, output_ptr, output.size(), op, launch.get_stream().get_stream());
  });
}

template <typename T>
static void cccl_argmax_new_segmented_reduce(nvbench::state& state, nvbench::type_list<T>)
{
  using offset_t = std::int64_t;

  const auto segments = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  auto segment_data   = make_random_0_to_5_segments<offset_t>(segments);

  thrust::device_vector<T> input = generate(segment_data.elements, bit_entropy::_1_000, T{0}, T{1});
  thrust::device_vector<argmax_pair<T>> pair_output(segments);
  thrust::device_vector<std::int64_t> output(segments);

  const auto* values       = thrust::raw_pointer_cast(input.data());
  const auto* local_id_ptr = thrust::raw_pointer_cast(segment_data.local_indices.data());
  const auto* offsets_ptr  = thrust::raw_pointer_cast(segment_data.offsets.data());
  auto* pairs_ptr          = thrust::raw_pointer_cast(pair_output.data());

  auto input_iter = cuda::make_transform_iterator(
    cuda::counting_iterator<offset_t>{0}, make_argmax_pair_op<T, offset_t>{values, local_id_ptr});

  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
    nullptr,
    temp_storage_bytes,
    input_iter,
    pairs_ptr,
    pair_output.size(),
    offsets_ptr,
    offsets_ptr + 1,
    argmax_pair_op<T>{},
    argmax_pair<T>{cuda::std::numeric_limits<T>::lowest(), -1});

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);

  auto extract_index = [] __device__(argmax_pair<T> item) -> std::int64_t {
    return item.local_index;
  };

  state.add_element_count(segment_data.elements);
  state.add_global_memory_reads<T>(segment_data.elements, "Size");
  state.add_global_memory_reads<offset_t>(segment_data.elements + segment_data.offsets.size());
  state.add_global_memory_writes<argmax_pair<T>>(segments);
  state.add_global_memory_writes<std::int64_t>(segments);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceSegmentedReduce::Reduce(
      thrust::raw_pointer_cast(temp_storage.data()),
      temp_storage_bytes,
      input_iter,
      pairs_ptr,
      pair_output.size(),
      offsets_ptr,
      offsets_ptr + 1,
      argmax_pair_op<T>{},
      argmax_pair<T>{cuda::std::numeric_limits<T>::lowest(), -1},
      launch.get_stream().get_stream());

    cub::DeviceTransform::Transform(
      pair_output.begin(), output.begin(), output.size(), extract_index, launch.get_stream().get_stream());
  });
}

using value_types = nvbench::type_list<double>;

NVBENCH_BENCH_TYPES(cccl_argmax_new_transform, NVBENCH_TYPE_AXES(value_types))
  .set_name("cccl_argmax_new_transform")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));

NVBENCH_BENCH_TYPES(cccl_argmax_new_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("cccl_argmax_new_segmented_reduce")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));
