// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cstdint>
#include <random>
#include <vector>

#include <nvbench_helper.cuh>

template <typename OffsetT>
struct small_segments
{
  thrust::device_vector<OffsetT> offsets;
  std::size_t elements{};
};

template <typename OffsetT>
static small_segments<OffsetT> make_random_0_to_5_segments(std::size_t segments)
{
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> length_dist(0, 5);

  std::vector<OffsetT> h_offsets(segments + 1);
  h_offsets[0] = 0;

  OffsetT total = 0;
  for (std::size_t segment = 0; segment < segments; ++segment)
  {
    const int length = length_dist(rng);
    total += static_cast<OffsetT>(length);
    h_offsets[segment + 1] = total;
  }

  return {thrust::device_vector<OffsetT>(h_offsets.begin(), h_offsets.end()), static_cast<std::size_t>(total)};
}

template <typename T, typename SegmentSize, std::size_t... Indices>
static auto make_fixed_size_transform_input(thrust::device_vector<T>& values, cuda::std::index_sequence<Indices...>)
{
  return cuda::std::make_tuple(cuda::make_strided_iterator(values.begin() + Indices, SegmentSize::value)...);
}

template <typename T, typename SegmentSize>
struct fixed_size_argmax_index_transform_op
{
  template <typename... Items>
  __device__ std::int64_t operator()(Items... items) const
  {
    static_assert(sizeof...(Items) == SegmentSize::value);
    T values[SegmentSize::value] = {static_cast<T>(items)...};
    T best_value                 = cuda::std::numeric_limits<T>::lowest();
    std::int64_t best_position   = -1;
#pragma unroll
    for (int local = 0; local < SegmentSize::value; ++local)
    {
      const T value = values[local];
      if (best_position == -1 || value > best_value || (value == best_value && local < best_position))
      {
        best_value    = value;
        best_position = static_cast<std::int64_t>(local);
      }
    }
    return best_position;
  }
};

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
      cuda::std::tuple{cuda::counting_iterator<offset_t>{0}},
      output_ptr,
      output.size(),
      op,
      launch.get_stream().get_stream());
  });
}

template <typename T, typename SegmentSize>
static void cccl_argmax_new_fixed_size_transform(nvbench::state& state, nvbench::type_list<T, SegmentSize>)
{
  constexpr int segment_size = SegmentSize::value;

  const auto segments = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  const auto elements = segments * segment_size;

  thrust::device_vector<T> values = generate(elements, bit_entropy::_1_000, T{0}, T{1});
  thrust::device_vector<std::int64_t> output(segments);

  auto input = make_fixed_size_transform_input<T, SegmentSize>(values, cuda::std::make_index_sequence<segment_size>{});

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<std::int64_t>(segments);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceTransform::Transform(
      input,
      output.begin(),
      output.size(),
      fixed_size_argmax_index_transform_op<T, SegmentSize>{},
      launch.get_stream().get_stream());
  });
}

template <typename T>
static void cccl_argmax_new_segmented_reduce(nvbench::state& state, nvbench::type_list<T>)
{
  using offset_t = std::int64_t;

  const auto segments = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  auto segment_data   = make_random_0_to_5_segments<offset_t>(segments);

  thrust::device_vector<T> input = generate(segment_data.elements, bit_entropy::_1_000, T{0}, T{1});
  thrust::device_vector<cub::KeyValuePair<int, T>> pair_output(segments);
  thrust::device_vector<std::int64_t> output(segments);

  const auto* offsets_ptr = thrust::raw_pointer_cast(segment_data.offsets.data());

  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMax(
    nullptr, temp_storage_bytes, input.begin(), pair_output.begin(), pair_output.size(), offsets_ptr, offsets_ptr + 1);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);

  auto extract_index = [] __device__(cub::KeyValuePair<int, T> item) -> std::int64_t {
    return item.key;
  };

  state.add_element_count(segment_data.elements);
  state.add_global_memory_reads<T>(segment_data.elements, "Size");
  state.add_global_memory_reads<offset_t>(segment_data.offsets.size());
  state.add_global_memory_writes<cub::KeyValuePair<int, T>>(segments);
  state.add_global_memory_writes<std::int64_t>(segments);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceSegmentedReduce::ArgMax(
      thrust::raw_pointer_cast(temp_storage.data()),
      temp_storage_bytes,
      input.begin(),
      pair_output.begin(),
      pair_output.size(),
      offsets_ptr,
      offsets_ptr + 1,
      launch.get_stream().get_stream());

    cub::DeviceTransform::Transform(
      cuda::std::tuple{pair_output.begin()},
      output.begin(),
      output.size(),
      extract_index,
      launch.get_stream().get_stream());
  });
}

using value_types = nvbench::type_list<double>;
using segment_size_types =
  nvbench::type_list<cuda::std::integral_constant<int, 2>,
                     cuda::std::integral_constant<int, 3>,
                     cuda::std::integral_constant<int, 4>,
                     cuda::std::integral_constant<int, 5>,
                     cuda::std::integral_constant<int, 6>>;

NVBENCH_BENCH_TYPES(cccl_argmax_new_transform, NVBENCH_TYPE_AXES(value_types))
  .set_name("cccl_argmax_new_transform")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));

NVBENCH_BENCH_TYPES(cccl_argmax_new_fixed_size_transform, NVBENCH_TYPE_AXES(value_types, segment_size_types))
  .set_name("cccl_argmax_new_fixed_size_transform")
  .set_type_axes_names({"T{ct}", "SegmentSize{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));

NVBENCH_BENCH_TYPES(cccl_argmax_new_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("cccl_argmax_new_segmented_reduce")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));
