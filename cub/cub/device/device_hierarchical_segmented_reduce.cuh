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

#include <cub/device/dispatch/dispatch_hierarchical_segmented_reduce.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

//! DeviceHierarchicalSegmentedReduce provides device-wide, fixed-size segmented
//! reductions whose cooperative epilog has access to the thread group, the
//! thread-local segment range, the segment output location, and the reduction
//! operator/init used for the segmented reduction.
struct DeviceHierarchicalSegmentedReduce
{
  template <typename InputIteratorT,
            typename SegmentOutputIteratorT,
            typename ReductionOpT,
            typename InitT,
            typename DeviceEpilogOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    InputIteratorT d_in,
    SegmentOutputIteratorT d_segment_out,
    ::cuda::std::int64_t num_segments,
    ::cuda::std::int64_t num_items,
    int segment_size,
    ReductionOpT reduction_op,
    InitT init,
    DeviceEpilogOpT device_epilog_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceHierarchicalSegmentedReduce::Reduce");

    return detail::hierarchical::dispatch_segmented_reduce(
      ::cuda::std::move(d_in),
      ::cuda::std::move(d_segment_out),
      num_segments,
      num_items,
      segment_size,
      ::cuda::std::move(reduction_op),
      ::cuda::std::move(init),
      ::cuda::std::move(device_epilog_op),
      stream);
  }
};

CUB_NAMESPACE_END
