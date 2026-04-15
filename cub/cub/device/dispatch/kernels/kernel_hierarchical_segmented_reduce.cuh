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

#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <int BlockThreads,
          typename InputIteratorT,
          typename SegmentOutputIteratorT,
          typename ReductionOpT,
          typename InitT,
          typename DeviceEpilogOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalSegmentedReduceKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const SegmentOutputIteratorT d_segment_out,
  _CCCL_GRID_CONSTANT const int num_segments,
  _CCCL_GRID_CONSTANT const ::cuda::std::int64_t num_items,
  _CCCL_GRID_CONSTANT const int segment_size,
  ReductionOpT reduction_op,
  InitT init,
  DeviceEpilogOpT device_epilog_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - each thread receives a contiguous slice of that segment via `thread_segment_range`
  // - the kernel computes a thread-local partial reduction over that slice using the segmented reduction operator
  // - the user epilog receives the thread group and thread-local partial aggregate, performs any cooperative combine
  //   it needs, and returns the materialized segment result while owning any additional side effects

  using partial_t = ::cuda::std::decay_t<decltype(::cuda::std::declval<ReductionOpT>()(
    ::cuda::std::declval<InitT>(), ::cuda::std::declval<cub::detail::it_reference_t<InputIteratorT>>()))>;

  const int segment_id = static_cast<int>(blockIdx.x);
  if (segment_id >= num_segments)
  {
    return;
  }

  const auto segment_offset =
    static_cast<::cuda::std::int64_t>(segment_id) * static_cast<::cuda::std::int64_t>(segment_size);

  if (segment_offset >= num_items)
  {
    return;
  }

  const int valid_items =
    static_cast<int>((::cuda::std::min) (static_cast<::cuda::std::int64_t>(segment_size), num_items - segment_offset));

  const auto block_hierarchy = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group           = ::cuda::experimental::this_block{block_hierarchy};
  auto input_range           = make_thread_segment_range<BlockThreads>(d_in + segment_offset, valid_items);
  partial_t thread_partial   = static_cast<partial_t>(init);

  for (int item = 0; item < input_range.size(); ++item)
  {
    thread_partial = reduction_op(thread_partial, input_range[item]);
  }

  const auto epilog_result = device_epilog_op(block_group, thread_partial);

  if (::cuda::gpu_thread.is_root_rank(block_group))
  {
    *(d_segment_out + segment_id) = epilog_result;
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
