// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load_to_shared.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/system/cuda/detail/core/util.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
struct transform_prolog_no_direct_input
{};

template <typename DirectInputIteratorT>
inline constexpr bool transform_prolog_has_direct_input_v =
  !::cuda::std::is_same_v<::cuda::std::remove_cv_t<DirectInputIteratorT>, transform_prolog_no_direct_input>;

using transform_prolog_f32_vector_storage_t = int4;
static_assert(sizeof(transform_prolog_f32_vector_storage_t) == 4 * sizeof(float));
static_assert(alignof(transform_prolog_f32_vector_storage_t) == 4 * sizeof(float));

template <typename DirectInputIteratorT>
struct transform_prolog_direct_vector_load
{
  static constexpr bool supported = false;
};

template <typename T>
struct transform_prolog_direct_vector_load<T*>
{
  using value_t = ::cuda::std::remove_cv_t<T>;

  static constexpr bool supported = ::cuda::std::is_same_v<value_t, float>;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static bool IsAligned(T* input, int index)
  {
    constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_f32_vector_storage_t));
    return (reinterpret_cast<::cuda::std::uintptr_t>(input + index) % alignment) == 0;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static transform_prolog_f32_vector_storage_t Load(T* input, int index)
  {
    return reinterpret_cast<const transform_prolog_f32_vector_storage_t*>(input + index)[0];
  }
};

template <CacheLoadModifier Modifier, typename OffsetT>
struct transform_prolog_direct_vector_load<CacheModifiedInputIterator<Modifier, float, OffsetT>>
{
  static constexpr bool supported = true;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static bool
  IsAligned(CacheModifiedInputIterator<Modifier, float, OffsetT> input, int index)
  {
    constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_f32_vector_storage_t));
    return (reinterpret_cast<::cuda::std::uintptr_t>(input.ptr + index) % alignment) == 0;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static transform_prolog_f32_vector_storage_t
  Load(CacheModifiedInputIterator<Modifier, float, OffsetT> input, int index)
  {
    CacheModifiedInputIterator<Modifier, transform_prolog_f32_vector_storage_t, OffsetT> vector_input(
      reinterpret_cast<transform_prolog_f32_vector_storage_t*>(input.ptr));
    return vector_input[index / 4];
  }
};

template <typename OutputIteratorT>
inline constexpr bool transform_prolog_vector_output_v =
  ::cuda::std::is_pointer_v<OutputIteratorT>
  && ::cuda::std::is_same_v<::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<OutputIteratorT>>, float>;

template <typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_prolog_store_vector(
  OutputIteratorT output, ::cuda::std::size_t index, transform_prolog_f32_vector_storage_t value)
{
  reinterpret_cast<transform_prolog_f32_vector_storage_t*>(output + index)[0] = value;
}

template <typename OutputIteratorT>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool
transform_prolog_output_vector_aligned(OutputIteratorT output, ::cuda::std::size_t index)
{
  constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_f32_vector_storage_t));
  return (reinterpret_cast<::cuda::std::uintptr_t>(output + index) % alignment) == 0;
}

template <typename ValueT>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE transform_prolog_f32_vector_storage_t
transform_prolog_load_shared_vector(ValueT* input, int index)
{
  return reinterpret_cast<const transform_prolog_f32_vector_storage_t*>(input + index)[0];
}

template <typename ValueT>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool transform_prolog_shared_vector_aligned(ValueT* input, int index)
{
  constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_f32_vector_storage_t));
  return (reinterpret_cast<::cuda::std::uintptr_t>(input + index) % alignment) == 0;
}

template <int BlockThreads,
          typename ValueT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentResultT,
          typename ApplyElementTransformT,
          typename ApplyElementTransformWithDirectT>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_prolog_store_outputs(
  const DirectInputIteratorT d_direct,
  const OutputIteratorT d_out,
  ::cuda::std::size_t segment_offset,
  int chunk_begin,
  int items,
  ValueT* shared_segment,
  const SegmentResultT& segment_result,
  ApplyElementTransformT apply_element_transform,
  ApplyElementTransformWithDirectT apply_element_transform_with_direct)
{
  const int thread_rank = static_cast<int>(threadIdx.x);

  if constexpr (transform_prolog_has_direct_input_v<DirectInputIteratorT>)
  {
    using direct_vector_load_t = transform_prolog_direct_vector_load<::cuda::std::remove_cv_t<DirectInputIteratorT>>;
    constexpr int vector_items = 4;
    constexpr bool can_vectorize =
      ::cuda::std::is_same_v<::cuda::std::remove_cv_t<ValueT>, float> && direct_vector_load_t::supported
      && transform_prolog_vector_output_v<::cuda::std::remove_cv_t<OutputIteratorT>>;

    if constexpr (can_vectorize)
    {
      const bool vector_aligned =
        (chunk_begin % vector_items == 0)
        && ((segment_offset + static_cast<::cuda::std::size_t>(chunk_begin)) % vector_items == 0)
        && direct_vector_load_t::IsAligned(d_direct, chunk_begin)
        && transform_prolog_shared_vector_aligned(shared_segment, 0)
        && transform_prolog_output_vector_aligned(d_out, segment_offset + static_cast<::cuda::std::size_t>(chunk_begin));

      if (vector_aligned)
      {
        const int vector_count = items / vector_items;
        for (int vector_index = thread_rank; vector_index < vector_count; vector_index += BlockThreads)
        {
          const int local_base       = vector_index * vector_items;
          const int index_in_segment = chunk_begin + local_base;

          using THRUST_NS_QUALIFIER::cuda_cub::core::detail::uninitialized_array;
          uninitialized_array<float, vector_items, sizeof(transform_prolog_f32_vector_storage_t)> direct_values;
          uninitialized_array<float, vector_items, sizeof(transform_prolog_f32_vector_storage_t)> shared_values;
          reinterpret_cast<transform_prolog_f32_vector_storage_t*>(direct_values.data())[0] =
            direct_vector_load_t::Load(d_direct, index_in_segment);
          reinterpret_cast<transform_prolog_f32_vector_storage_t*>(shared_values.data())[0] =
            transform_prolog_load_shared_vector(shared_segment, local_base);

          uninitialized_array<float, vector_items, sizeof(transform_prolog_f32_vector_storage_t)> transformed_values;
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int lane = 0; lane < vector_items; ++lane)
          {
            transformed_values[lane] =
              apply_element_transform_with_direct(segment_result, direct_values[lane], shared_values[lane]);
          }

          transform_prolog_store_vector(
            d_out,
            segment_offset + static_cast<::cuda::std::size_t>(index_in_segment),
            reinterpret_cast<const transform_prolog_f32_vector_storage_t*>(transformed_values.data())[0]);
        }

        for (int local_index = vector_count * vector_items + thread_rank; local_index < items;
             local_index += BlockThreads)
        {
          const int index_in_segment = chunk_begin + local_index;
          const auto direct_value    = d_direct[index_in_segment];
          d_out[segment_offset + static_cast<::cuda::std::size_t>(index_in_segment)] =
            apply_element_transform_with_direct(segment_result, direct_value, shared_segment[local_index]);
        }
        return;
      }
    }

    for (int local_index = thread_rank; local_index < items; local_index += BlockThreads)
    {
      const int index_in_segment = chunk_begin + local_index;
      const auto direct_value    = d_direct[index_in_segment];
      d_out[segment_offset + static_cast<::cuda::std::size_t>(index_in_segment)] =
        apply_element_transform_with_direct(segment_result, direct_value, shared_segment[local_index]);
    }
  }
  else
  {
    for (int local_index = thread_rank; local_index < items; local_index += BlockThreads)
    {
      const int index_in_segment = chunk_begin + local_index;
      d_out[segment_offset + static_cast<::cuda::std::size_t>(index_in_segment)] =
        apply_element_transform(segment_result, index_in_segment, shared_segment[local_index]);
    }
  }
}

template <int BlockThreads,
          int BulkCopyAlignment,
          typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const DirectInputIteratorT d_direct,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int segment_size,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - the block first stages the segment into shared memory via `BlockLoadToShared`
  // - `segment_op` receives either an explicitly requested contiguous span or an implementation-chosen per-thread range
  // - `segment_op` is responsible for any block-wide combine it needs and should return the final segment result on
  //   every thread in the block group
  // - the kernel then applies `element_transform_op` to each item, passing the segment result, the segment-local item
  //   index, and the item value

  using block_hierarchy_t = decltype(::cuda::hierarchy(::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using block_group_t     = ::cuda::experimental::this_block<block_hierarchy_t>;
  using value_t           = cub::detail::it_value_t<InputIteratorT>;
  using input_range_t     = transform_prolog_segment_range_t<BlockThreads, block_group_t, SegmentOpT, value_t>;
  using segment_result_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, block_group_t, input_range_t>>;
  using block_load_to_shared_t = BlockLoadToShared<BlockThreads>;

  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  extern __shared__ char shared_segment_buffer_base[];
  __shared__ typename block_load_to_shared_t::TempStorage load_to_shared_storage;

  const int segment_id = static_cast<int>(blockIdx.x);
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const auto segment_begin = d_in + segment_offset;

  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group             = ::cuda::experimental::this_block{block_hierarchy};
  auto apply_element_transform = [&](const segment_result_t& segment_result, int index_in_segment, auto&& value) {
    using input_ref_t = decltype(value);

    if constexpr (::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, int, input_ref_t>)
    {
      return element_transform_op(segment_result, index_in_segment, static_cast<input_ref_t>(value));
    }
    else
    {
      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, input_ref_t>,
                    "element_transform_op must be invocable with either "
                    "(segment_result, index_in_segment, value) or (segment_result, value).");
      return element_transform_op(segment_result, static_cast<input_ref_t>(value));
    }
  };
  auto apply_element_transform_with_direct =
    [&](const segment_result_t& segment_result, auto&& direct_value, auto&& value) {
      using direct_ref_t = decltype(direct_value);
      using input_ref_t  = decltype(value);

      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, direct_ref_t, input_ref_t>,
                    "element_transform_op must be invocable with (segment_result, direct_input, value).");
      return element_transform_op(
        segment_result, static_cast<direct_ref_t>(direct_value), static_cast<input_ref_t>(value));
    };

  const int shared_buffer_bytes = cub::detail::LoadToSharedBufferSizeBytes<value_t, BulkCopyAlignment>(segment_size);

  block_load_to_shared_t load_to_shared{load_to_shared_storage};
  ::cuda::std::span<char> shared_buffer{
    shared_segment_buffer_base, static_cast<::cuda::std::size_t>(shared_buffer_bytes)};
  ::cuda::std::span<const value_t> input_buffer{segment_begin, static_cast<::cuda::std::size_t>(segment_size)};
  auto staged_segment = load_to_shared.template CopyAsync<value_t, BulkCopyAlignment>(shared_buffer, input_buffer);
  auto token          = load_to_shared.Commit();
  load_to_shared.Wait(::cuda::std::move(token));
  value_t* shared_segment = staged_segment.data();

  auto input_range =
    make_transform_prolog_segment_range<BlockThreads, block_group_t, SegmentOpT>(shared_segment, segment_size);
  const segment_result_t segment_result = segment_op(block_group, input_range);

  transform_prolog_store_outputs<BlockThreads>(
    d_direct,
    d_out,
    segment_offset,
    0,
    segment_size,
    shared_segment,
    segment_result,
    apply_element_transform,
    apply_element_transform_with_direct);
}

template <int BlockThreads,
          int BulkCopyAlignment,
          typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformClusterKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const DirectInputIteratorT d_direct,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int segment_size,
  _CCCL_GRID_CONSTANT const int cluster_size,
  _CCCL_GRID_CONSTANT const int chunk_items,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Large-segment path:
  // - one cluster owns one fixed-size contiguous segment
  // - each CTA in the cluster stages one contiguous segment chunk in its local shared memory
  // - the segment op sees a cluster group and the CTA-local chunk range
  // - the segment op is responsible for reducing across the cluster, for example through cuda::coop::reduce
  // - each CTA transforms and stores only the items from its own staged chunk

  using block_hierarchy_t = decltype(::cuda::hierarchy(::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using cluster_hierarchy_t = decltype(::cuda::hierarchy(
    ::cuda::grid_dims(dim3{}), ::cuda::cluster_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using cluster_group_t     = ::cuda::experimental::this_cluster<cluster_hierarchy_t>;
  using value_t             = cub::detail::it_value_t<InputIteratorT>;
  using input_range_t       = transform_prolog_segment_range_t<BlockThreads, cluster_group_t, SegmentOpT, value_t>;
  using segment_result_t =
    ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, cluster_group_t, input_range_t>>;
  using block_load_to_shared_t = BlockLoadToShared<BlockThreads>;

  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");

  extern __shared__ char shared_segment_buffer_base[];
  __shared__ typename block_load_to_shared_t::TempStorage load_to_shared_storage;

  const int cluster_rank = static_cast<int>(blockIdx.x % cluster_size);
  const int segment_id   = static_cast<int>(blockIdx.x / cluster_size);
  const int chunk_begin  = (::cuda::std::min) (cluster_rank * chunk_items, segment_size);
  const int chunk_end    = (::cuda::std::min) (chunk_begin + chunk_items, segment_size);
  const int local_items  = chunk_end - chunk_begin;
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const auto segment_begin = d_in + segment_offset;

  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group             = ::cuda::experimental::this_block{block_hierarchy};
  const auto cluster_hierarchy = ::cuda::hierarchy(
    ::cuda::grid_dims(gridDim),
    ::cuda::cluster_dims(dim3(static_cast<unsigned int>(cluster_size), 1, 1)),
    ::cuda::block_dims<BlockThreads>());
  auto cluster_group = ::cuda::experimental::this_cluster{cluster_hierarchy};

  auto apply_element_transform = [&](const segment_result_t& segment_result, int index_in_segment, auto&& value) {
    using input_ref_t = decltype(value);

    if constexpr (::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, int, input_ref_t>)
    {
      return element_transform_op(segment_result, index_in_segment, static_cast<input_ref_t>(value));
    }
    else
    {
      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, input_ref_t>,
                    "element_transform_op must be invocable with either "
                    "(segment_result, index_in_segment, value) or (segment_result, value).");
      return element_transform_op(segment_result, static_cast<input_ref_t>(value));
    }
  };
  auto apply_element_transform_with_direct =
    [&](const segment_result_t& segment_result, auto&& direct_value, auto&& value) {
      using direct_ref_t = decltype(direct_value);
      using input_ref_t  = decltype(value);

      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, direct_ref_t, input_ref_t>,
                    "element_transform_op must be invocable with (segment_result, direct_input, value).");
      return element_transform_op(
        segment_result, static_cast<direct_ref_t>(direct_value), static_cast<input_ref_t>(value));
    };

  const int shared_buffer_bytes = cub::detail::LoadToSharedBufferSizeBytes<value_t, BulkCopyAlignment>(local_items);

  block_load_to_shared_t load_to_shared{load_to_shared_storage};
  ::cuda::std::span<char> shared_buffer{
    shared_segment_buffer_base, static_cast<::cuda::std::size_t>(shared_buffer_bytes)};
  ::cuda::std::span<const value_t> input_buffer{
    segment_begin + chunk_begin, static_cast<::cuda::std::size_t>(local_items)};
  auto staged_segment = load_to_shared.template CopyAsync<value_t, BulkCopyAlignment>(shared_buffer, input_buffer);
  auto token          = load_to_shared.Commit();
  load_to_shared.Wait(::cuda::std::move(token));
  value_t* shared_segment = staged_segment.data();

  auto input_range = make_transform_prolog_segment_range<BlockThreads, cluster_group_t, SegmentOpT>(
    shared_segment, local_items, chunk_begin);
  const segment_result_t segment_result = segment_op(cluster_group, input_range);

  transform_prolog_store_outputs<BlockThreads>(
    d_direct,
    d_out,
    segment_offset,
    chunk_begin,
    local_items,
    shared_segment,
    segment_result,
    apply_element_transform,
    apply_element_transform_with_direct);
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
