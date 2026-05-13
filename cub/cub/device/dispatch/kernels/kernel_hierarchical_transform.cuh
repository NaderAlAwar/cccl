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

#include <cuda/__cmath/round_up.h>
#include <cuda/__memory/align_down.h>
#include <cuda/std/cstddef>
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

constexpr int transform_prolog_bulk_copy_alignment =
  (CUB_PTX_ARCH >= 900 && CUB_PTX_ARCH < 1000) ? 128 : cub::detail::bulk_copy_min_align;

_CCCL_HOST_DEVICE constexpr int transform_prolog_bulk_copy_alignment_for_ptx(int ptx_version)
{
  return (ptx_version >= 900 && ptx_version < 1000) ? 128 : cub::detail::bulk_copy_min_align;
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_alignment(int bulk_copy_alignment)
{
  constexpr int buffer_alignment = cub::detail::LoadToSharedBufferAlignBytes<T>();
  return bulk_copy_alignment > buffer_alignment ? bulk_copy_alignment : buffer_alignment;
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_alignment()
{
  return transform_prolog_load_to_shared_buffer_alignment<T>(transform_prolog_bulk_copy_alignment);
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_size(int items, int bulk_copy_alignment)
{
  if (items == 0)
  {
    return 0;
  }

  const auto payload_bytes   = static_cast<::cuda::std::size_t>(items) * static_cast<::cuda::std::size_t>(sizeof(T));
  const int max_head_padding = bulk_copy_alignment - 1;
  return cub::detail::LoadToSharedBufferSizeBytes<char>(payload_bytes + max_head_padding);
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_size(int items)
{
  return transform_prolog_load_to_shared_buffer_size<T>(items, transform_prolog_bulk_copy_alignment);
}

_CCCL_DEVICE _CCCL_FORCEINLINE const char*
transform_prolog_align_copy_source(const char* source, ::cuda::std::size_t bytes_before_source, int& head_padding)
{
  const char* const aligned_source = ::cuda::align_down(source, transform_prolog_bulk_copy_alignment);
  head_padding                     = static_cast<int>(source - aligned_source);
  if (static_cast<::cuda::std::size_t>(head_padding) <= bytes_before_source)
  {
    return aligned_source;
  }

  head_padding = 0;
  return source;
}

template <int Alignment>
_CCCL_DEVICE _CCCL_FORCEINLINE char* align_dynamic_shared_buffer(char* shared_buffer_base)
{
  if constexpr (Alignment > 16)
  {
    uint32_t shared_buffer_ptr = __cvta_generic_to_shared(shared_buffer_base);
    shared_buffer_ptr          = ::cuda::round_up(shared_buffer_ptr, static_cast<uint32_t>(Alignment));
    asm("" : "+r"(shared_buffer_ptr));
    return static_cast<char*>(__cvta_shared_to_generic(shared_buffer_ptr));
  }
  else
  {
    return shared_buffer_base;
  }
}

template <int BlockThreads,
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
  _CCCL_GRID_CONSTANT const int items_per_thread,
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

  // There is a subtle difference with the epilog case. There we did
  // IPT because each thread was doing ipt items. Here each thread
  // does do IPT items but it also loads other stuff so it can
  // calculate the RMS. So here, BlockLoad does not need to be tied to
  // IPT in the same way.
  const int tile_items = BlockThreads * items_per_thread;

  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  extern __shared__ char shared_segment_buffer_base[];
  __shared__ typename block_load_to_shared_t::TempStorage load_to_shared_storage;

  const int segment_id = static_cast<int>(blockIdx.x);
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const auto segment_begin = d_in + segment_offset;
  const int thread_rank    = static_cast<int>(threadIdx.x);

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

  constexpr int shared_buffer_alignment = transform_prolog_load_to_shared_buffer_alignment<value_t>();
  char* aligned_shared_buffer     = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
  const char* const segment_src   = reinterpret_cast<const char*>(segment_begin);
  const char* aligned_segment_src = segment_src;
  int source_head_padding         = 0;
  int bytes_to_copy               = 0;

  // Align the bulk copy source to the preferred boundary, then ignore the copied head bytes by shifting the shared
  // pointer back to the logical segment.
  if (segment_size > 0)
  {
    aligned_segment_src =
      transform_prolog_align_copy_source(segment_src, segment_offset * sizeof(value_t), source_head_padding);
    bytes_to_copy = source_head_padding + segment_size * static_cast<int>(sizeof(value_t));
  }

  const int shared_buffer_bytes = cub::detail::LoadToSharedBufferSizeBytes<char>(bytes_to_copy);

  block_load_to_shared_t load_to_shared{load_to_shared_storage};
  ::cuda::std::span<char> shared_buffer{aligned_shared_buffer, static_cast<::cuda::std::size_t>(shared_buffer_bytes)};
  ::cuda::std::span<const char> input_buffer{aligned_segment_src, static_cast<::cuda::std::size_t>(bytes_to_copy)};
  auto staged_bytes = load_to_shared.CopyAsync(shared_buffer, input_buffer);
  auto token        = load_to_shared.Commit();
  load_to_shared.Wait(::cuda::std::move(token));
  value_t* shared_segment = reinterpret_cast<value_t*>(staged_bytes.data() + source_head_padding);

  auto input_range =
    make_transform_prolog_segment_range<BlockThreads, block_group_t, SegmentOpT>(shared_segment, segment_size);
  const segment_result_t segment_result = segment_op(block_group, input_range);

  for (int tile_base = 0; tile_base < segment_size; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, segment_size - tile_base);

    for (int item = 0; item < items_per_thread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;

      if (tile_local_index < valid_items)
      {
        const int index_in_segment = tile_base + tile_local_index;
        if constexpr (transform_prolog_has_direct_input_v<DirectInputIteratorT>)
        {
          const auto direct_value = d_direct[index_in_segment];
          d_out[segment_offset + index_in_segment] =
            apply_element_transform_with_direct(segment_result, direct_value, shared_segment[index_in_segment]);
        }
        else
        {
          d_out[segment_offset + index_in_segment] =
            apply_element_transform(segment_result, index_in_segment, shared_segment[index_in_segment]);
        }
      }
    }
  }
}

template <int BlockThreads,
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
  _CCCL_GRID_CONSTANT const int items_per_thread,
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

  const int tile_items = BlockThreads * items_per_thread;

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
  const int thread_rank    = static_cast<int>(threadIdx.x);

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

  constexpr int shared_buffer_alignment = transform_prolog_load_to_shared_buffer_alignment<value_t>();
  char* aligned_shared_buffer   = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
  const char* const chunk_src   = reinterpret_cast<const char*>(segment_begin + chunk_begin);
  const char* aligned_chunk_src = chunk_src;
  int source_head_padding       = 0;
  int bytes_to_copy             = 0;

  // Align the bulk copy source to the preferred boundary, then ignore the copied head bytes by shifting the shared
  // pointer back to the logical chunk.
  if (local_items > 0)
  {
    const auto bytes_before_chunk = (segment_offset + static_cast<::cuda::std::size_t>(chunk_begin)) * sizeof(value_t);
    aligned_chunk_src     = transform_prolog_align_copy_source(chunk_src, bytes_before_chunk, source_head_padding);
    const int chunk_bytes = local_items * static_cast<int>(sizeof(value_t));
    bytes_to_copy         = source_head_padding + chunk_bytes;
  }

  const int shared_buffer_bytes = cub::detail::LoadToSharedBufferSizeBytes<char>(bytes_to_copy);

  block_load_to_shared_t load_to_shared{load_to_shared_storage};
  ::cuda::std::span<char> shared_buffer{aligned_shared_buffer, static_cast<::cuda::std::size_t>(shared_buffer_bytes)};
  ::cuda::std::span<const char> input_buffer{aligned_chunk_src, static_cast<::cuda::std::size_t>(bytes_to_copy)};
  auto staged_bytes = load_to_shared.CopyAsync(shared_buffer, input_buffer);
  auto token        = load_to_shared.Commit();
  load_to_shared.Wait(::cuda::std::move(token));
  value_t* shared_segment = reinterpret_cast<value_t*>(staged_bytes.data() + source_head_padding);

  auto input_range = make_transform_prolog_segment_range<BlockThreads, cluster_group_t, SegmentOpT>(
    shared_segment, local_items, chunk_begin);
  const segment_result_t segment_result = segment_op(cluster_group, input_range);

  for (int tile_base = 0; tile_base < local_items; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, local_items - tile_base);

    for (int item = 0; item < items_per_thread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;

      if (tile_local_index < valid_items)
      {
        const int local_index      = tile_base + tile_local_index;
        const int index_in_segment = chunk_begin + local_index;
        if constexpr (transform_prolog_has_direct_input_v<DirectInputIteratorT>)
        {
          const auto direct_value = d_direct[index_in_segment];
          d_out[segment_offset + index_in_segment] =
            apply_element_transform_with_direct(segment_result, direct_value, shared_segment[local_index]);
        }
        else
        {
          d_out[segment_offset + index_in_segment] =
            apply_element_transform(segment_result, index_in_segment, shared_segment[local_index]);
        }
      }
    }
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
