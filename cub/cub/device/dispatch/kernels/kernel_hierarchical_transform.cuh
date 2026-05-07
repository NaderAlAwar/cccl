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

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
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
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int segment_size,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - each thread receives a contiguous slice of that segment via `thread_segment_range`
  // - `segment_op` is responsible for any block-wide combine it needs and should return the final segment result on
  //   every thread in the block group
  // - the block first stages the segment into shared memory via tiled `BlockLoad`
  // - the kernel then applies `element_transform_op` to each item, passing the segment result, the segment-local item
  //   index, and the item value

  using block_hierarchy_t = decltype(::cuda::hierarchy(::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using block_group_t     = ::cuda::experimental::this_block<block_hierarchy_t>;
  using value_t           = cub::detail::it_value_t<InputIteratorT>;
  using input_range_t     = thread_segment_range<value_t*>;
  using segment_result_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, block_group_t, input_range_t>>;
  using block_load_t     = BlockLoad<value_t, BlockThreads, ItemsPerThread, BLOCK_LOAD_STRIPED>;
  using block_store_t    = BlockStore<value_t, BlockThreads, ItemsPerThread, BLOCK_STORE_STRIPED>;

  // There is a subtle difference with the epilog case. There we did
  // IPT because each thread was doing ipt items. Here each thread
  // does do IPT items but it also loads other stuff so it cal
  // calculate the RMS. So here, BlockLoad does not need to be tied to
  // IPT in the same way.
  constexpr int tile_items = BlockThreads * ItemsPerThread;

  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  extern __shared__ char shared_segment_buffer_base[];

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

  constexpr int shared_buffer_alignment = alignof(value_t);
  char* aligned_shared_buffer = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
  value_t* shared_segment     = reinterpret_cast<value_t*>(aligned_shared_buffer);

  for (int tile_base = 0; tile_base < segment_size; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, segment_size - tile_base);
    value_t items[ItemsPerThread];

    // TODO: very important, consider using block load to shared for
    // speedups.

    block_load_t().Load(segment_begin + tile_base, items, valid_items);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;

      if (tile_local_index < valid_items)
      {
        shared_segment[tile_base + tile_local_index] = items[item];
      }
    }
  }

  block_group.sync();

  auto input_range                      = make_thread_segment_range<BlockThreads>(shared_segment, segment_size);
  const segment_result_t segment_result = segment_op(block_group, input_range);

  for (int tile_base = 0; tile_base < segment_size; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, segment_size - tile_base);
    value_t output_items[ItemsPerThread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;
      output_items[item]         = value_t{};

      if (tile_local_index < valid_items)
      {
        const int index_in_segment = tile_base + tile_local_index;
        output_items[item] =
          apply_element_transform(segment_result, index_in_segment, shared_segment[index_in_segment]);
      }
    }

    block_store_t().Store(d_out + segment_offset + tile_base, output_items, valid_items);
  }
}

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformClusterKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
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
  // - the segment op is responsible for reducing across the cluster, for example through cuda::device::reduce
  // - each CTA transforms and stores only the items from its own staged chunk

  using block_hierarchy_t = decltype(::cuda::hierarchy(::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using cluster_hierarchy_t = decltype(::cuda::hierarchy(
    ::cuda::grid_dims(dim3{}), ::cuda::cluster_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using cluster_group_t     = ::cuda::experimental::this_cluster<cluster_hierarchy_t>;
  using value_t             = cub::detail::it_value_t<InputIteratorT>;
  using input_range_t       = thread_segment_range<value_t*>;
  using segment_result_t =
    ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, cluster_group_t, input_range_t>>;
  using block_load_t  = BlockLoad<value_t, BlockThreads, ItemsPerThread, BLOCK_LOAD_STRIPED>;
  using block_store_t = BlockStore<value_t, BlockThreads, ItemsPerThread, BLOCK_STORE_STRIPED>;

  constexpr int tile_items = BlockThreads * ItemsPerThread;

  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");

  extern __shared__ char shared_segment_buffer_base[];

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

  constexpr int shared_buffer_alignment = alignof(value_t);
  char* aligned_shared_buffer = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
  value_t* shared_segment     = reinterpret_cast<value_t*>(aligned_shared_buffer);

  for (int tile_base = 0; tile_base < local_items; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, local_items - tile_base);
    value_t items[ItemsPerThread];

    block_load_t().Load(segment_begin + chunk_begin + tile_base, items, valid_items);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;

      if (tile_local_index < valid_items)
      {
        shared_segment[tile_base + tile_local_index] = items[item];
      }
    }
  }

  block_group.sync();

  const auto local_range = make_thread_segment_range<BlockThreads>(shared_segment, local_items);
  auto input_range       = input_range_t{local_range.begin(), chunk_begin + local_range.offset(), local_range.size()};
  const segment_result_t segment_result = segment_op(cluster_group, input_range);

  for (int tile_base = 0; tile_base < local_items; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, local_items - tile_base);
    value_t output_items[ItemsPerThread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;
      output_items[item]         = value_t{};

      if (tile_local_index < valid_items)
      {
        const int local_index      = tile_base + tile_local_index;
        const int index_in_segment = chunk_begin + local_index;
        output_items[item] = apply_element_transform(segment_result, index_in_segment, shared_segment[local_index]);
      }
    }

    block_store_t().Store(d_out + segment_offset + chunk_begin + tile_base, output_items, valid_items);
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
