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

#ifndef _CUDAX_HIERARCHY
#  define _CUDAX_HIERARCHY
#endif

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/hierarchy>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/hierarchy.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <typename InputIteratorT>
inline constexpr bool shared_input_staging_supported_v =
  ::cuda::std::is_trivially_copyable_v<cub::detail::it_value_t<InputIteratorT>>;
} // namespace detail::hierarchical

namespace hierarchical
{
template <typename Hierarchy, typename T, typename ReductionOp>
_CCCL_DEVICE T reduce(::cuda::experimental::this_block<Hierarchy> /*group*/, T thread_data, ReductionOp reduction_op)
{
  using block_extents_t = decltype(::cuda::gpu_thread.extents(::cuda::block, ::cuda::std::declval<Hierarchy>()));

  static_assert(block_extents_t::rank_dynamic() == 0,
                "cub::hierarchical::reduce currently requires statically sized block groups.");

  using collective_t =
    BlockReduce<T,
                static_cast<int>(block_extents_t::static_extent(0)),
                cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                static_cast<int>(block_extents_t::static_extent(1)),
                static_cast<int>(block_extents_t::static_extent(2))>;

  return collective_t{}.Reduce(thread_data, reduction_op);
}

template <typename Hierarchy, typename T>
_CCCL_DEVICE T reduce(::cuda::experimental::this_block<Hierarchy> group, T thread_data)
{
  return reduce(group, thread_data, ::cuda::std::plus<>{});
}
} // namespace hierarchical

namespace detail::hierarchical
{
template <typename RandomAccessIteratorT>
class thread_segment_range
{
public:
  using iterator       = RandomAccessIteratorT;
  using const_iterator = RandomAccessIteratorT;
  using value_type     = cub::detail::it_value_t<RandomAccessIteratorT>;
  using reference      = cub::detail::it_reference_t<RandomAccessIteratorT>;

  _CCCL_HOST_DEVICE constexpr thread_segment_range() noexcept = default;

  _CCCL_HOST_DEVICE constexpr thread_segment_range(iterator begin, int offset, int items) noexcept
      : begin_(begin)
      , offset_(offset)
      , items_(items)
  {}

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr iterator begin() const noexcept
  {
    return begin_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr iterator end() const noexcept
  {
    return begin_ + items_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int offset() const noexcept
  {
    return offset_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int size() const noexcept
  {
    return items_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr bool empty() const noexcept
  {
    return items_ == 0;
  }

  _CCCL_HOST_DEVICE constexpr reference operator[](int idx) const noexcept
  {
    return begin_[idx];
  }

private:
  iterator begin_{};
  int offset_{0};
  int items_{0};
};

template <int BlockThreads, typename RandomAccessIteratorT>
_CCCL_DEVICE auto make_thread_segment_range(RandomAccessIteratorT segment_begin, int segment_size)
{
  const int thread_rank   = static_cast<int>(threadIdx.x);
  const int base_items    = segment_size / BlockThreads;
  const int remainder     = segment_size % BlockThreads;
  const int thread_items  = base_items + (thread_rank < remainder ? 1 : 0);
  const int thread_offset = thread_rank * base_items + ((thread_rank < remainder) ? thread_rank : remainder);

  return thread_segment_range<RandomAccessIteratorT>{segment_begin + thread_offset, thread_offset, thread_items};
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
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int num_segments,
  _CCCL_GRID_CONSTANT const int segment_size,
  _CCCL_GRID_CONSTANT const bool use_shared_input_staging,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - each thread receives a contiguous slice of that segment via `thread_segment_range`
  // - `segment_op` is responsible for any block-wide combine it needs and should return the final segment result on
  //   the root thread (or redundantly on every thread)
  // - when enabled, the block first stages the segment into shared memory via tiled `BlockLoad`
  // - the kernel broadcasts the root result to the whole block and then applies `element_transform_op` to each item,
  //   passing the segment result, the segment-local item index, and the item value

  using segment_result_t = typename SegmentOpT::result_type;
  using value_t          = cub::detail::it_value_t<InputIteratorT>;
  using block_load_t     = BlockLoad<value_t, BlockThreads, ItemsPerThread, BLOCK_LOAD_STRIPED>;
  using block_store_t    = BlockStore<value_t, BlockThreads, ItemsPerThread, BLOCK_STORE_STRIPED>;

  constexpr int tile_items = BlockThreads * ItemsPerThread;

  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  static_assert(::cuda::std::is_trivially_copyable_v<segment_result_t>,
                "The current hierarchical kernel stores the segment result in shared memory.");

  __shared__ segment_result_t segment_result;
  extern __shared__ char shared_segment_buffer_base[];

  const int segment_id = static_cast<int>(blockIdx.x);
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const auto segment_begin = d_in + segment_offset;
  const int thread_rank    = static_cast<int>(threadIdx.x);

  const auto block_hierarchy = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group           = ::cuda::experimental::this_block{block_hierarchy};
  auto output_range          = make_thread_segment_range<BlockThreads>(d_out + segment_offset, segment_size);

  auto transform_segment = [&](auto input_range) {
    const segment_result_t thread_result = segment_op(block_group, input_range);

    if (::cuda::gpu_thread.is_root_rank(block_group))
    {
      segment_result = thread_result;
    }

    block_group.sync();

    for (int item_idx = 0; item_idx < input_range.size(); ++item_idx)
    {
      const int index_in_segment = input_range.offset() + item_idx;
      output_range[item_idx]     = element_transform_op(segment_result, index_in_segment, input_range[item_idx]);
    }
  };

  if constexpr (shared_input_staging_supported_v<InputIteratorT>)
  {
    if (use_shared_input_staging)
    {
      constexpr int shared_buffer_alignment = alignof(value_t);
      char* aligned_shared_buffer = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
      value_t* shared_segment     = reinterpret_cast<value_t*>(aligned_shared_buffer);

      for (int tile_base = 0; tile_base < segment_size; tile_base += tile_items)
      {
        const int valid_items = (::cuda::std::min) (tile_items, segment_size - tile_base);
        value_t items[ItemsPerThread];

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

      auto input_range                     = make_thread_segment_range<BlockThreads>(shared_segment, segment_size);
      const segment_result_t thread_result = segment_op(block_group, input_range);

      if (::cuda::gpu_thread.is_root_rank(block_group))
      {
        segment_result = thread_result;
      }

      block_group.sync();

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
              element_transform_op(segment_result, index_in_segment, shared_segment[index_in_segment]);
          }
        }

        block_store_t().Store(d_out + segment_offset + tile_base, output_items, valid_items);
      }

      return;
    }
  }

  transform_segment(make_thread_segment_range<BlockThreads>(segment_begin, segment_size));
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
