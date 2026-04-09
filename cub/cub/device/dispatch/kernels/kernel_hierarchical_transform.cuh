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

#include <cub/util_type.cuh>

#include <cuda/hierarchy>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/hierarchy.cuh>

CUB_NAMESPACE_BEGIN

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

template <int BlockThreads,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int num_segments,
  _CCCL_GRID_CONSTANT const int segment_size,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - each thread receives a contiguous slice of that segment via `thread_segment_range`
  // - `segment_op` is responsible for any block-wide combine it needs and should return the final segment result on
  //   the root thread (or redundantly on every thread)
  // - the kernel broadcasts the root result to the whole block and then applies `element_transform_op`
  const int segment_id = static_cast<int>(blockIdx.x);
  if (segment_id >= num_segments)
  {
    return;
  }

  using segment_result_t = typename SegmentOpT::result_type;

  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  static_assert(::cuda::std::is_trivially_copyable_v<segment_result_t>,
                "The current hierarchical kernel stores the segment result in shared memory.");

  __shared__ segment_result_t segment_result;

  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);

  using block_group_t = ::cuda::experimental::this_block<::cuda::experimental::__implicit_hierarchy_t>;
  block_group_t block_group{};
  auto input_range  = make_thread_segment_range<BlockThreads>(d_in + segment_offset, segment_size);
  auto output_range = make_thread_segment_range<BlockThreads>(d_out + segment_offset, segment_size);

  const segment_result_t thread_result = segment_op(block_group, input_range);

  if (::cuda::gpu_thread.is_root_rank(block_group))
  {
    segment_result = thread_result;
  }

  block_group.sync();

  for (int item_idx = 0; item_idx < input_range.size(); ++item_idx)
  {
    output_range[item_idx] = element_transform_op(segment_result, input_range[item_idx]);
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
