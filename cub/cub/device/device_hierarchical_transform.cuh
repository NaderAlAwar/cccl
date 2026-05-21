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
#include <cub/device/dispatch/dispatch_hierarchical_transform_epilog.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

//! DeviceSegmentedTransform provides device-wide, fixed-size segmented
//! hierarchical transform operations.
struct DeviceSegmentedTransform
{
  template <typename InputIteratorT,
            typename DirectInputIteratorT,
            typename OutputIteratorT,
            typename SegmentOpT,
            typename ElementTransformOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformProlog(
    InputIteratorT d_in,
    DirectInputIteratorT d_direct,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    SegmentOpT segment_op,
    ElementTransformOpT element_transform_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedTransform::TransformProlog");

    return detail::hierarchical::dispatch(
      ::cuda::std::move(d_in),
      ::cuda::std::move(d_direct),
      ::cuda::std::move(d_out),
      num_segments,
      segment_size,
      ::cuda::std::move(segment_op),
      ::cuda::std::move(element_transform_op),
      stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT, typename TransformOpT, typename DeviceEpilogOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformEpilog(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    TransformOpT transform_op,
    DeviceEpilogOpT device_epilog_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedTransform::TransformEpilog");

    return detail::hierarchical::dispatch_transform_epilog(
      ::cuda::std::move(d_in),
      ::cuda::std::move(d_out),
      num_segments,
      segment_size,
      ::cuda::std::move(transform_op),
      ::cuda::std::move(device_epilog_op),
      stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT, typename TransformOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformEpilog(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    TransformOpT transform_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedTransform::TransformEpilog");

    return detail::hierarchical::dispatch_transform_epilog(
      ::cuda::std::move(d_in),
      ::cuda::std::move(d_out),
      num_segments,
      segment_size,
      ::cuda::std::move(transform_op),
      stream);
  }
};

CUB_NAMESPACE_END
