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

#include <cub/device/dispatch/dispatch_hierarchical_transform.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

//! DeviceHierarchicalTransform provides device-wide, fixed-size segmented
//! reduce-then-transform operations.
struct DeviceHierarchicalTransform
{
  template <typename InputIteratorT, typename OutputIteratorT, typename SegmentOpT, typename ElementTransformOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    SegmentOpT segment_op,
    ElementTransformOpT element_transform_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceHierarchicalTransform::Transform");

    return detail::hierarchical::dispatch(
      ::cuda::std::move(d_in),
      ::cuda::std::move(d_out),
      num_segments,
      segment_size,
      ::cuda::std::move(segment_op),
      ::cuda::std::move(element_transform_op),
      stream);
  }
};

CUB_NAMESPACE_END
