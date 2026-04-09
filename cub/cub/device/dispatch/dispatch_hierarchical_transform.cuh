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

#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_transform.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <int BlockThreads,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
struct DeviceHierarchicalTransformKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformKernel,
    DeviceHierarchicalTransformKernel<BlockThreads, InputIteratorT, OutputIteratorT, SegmentOpT, ElementTransformOpT>)
};

template <
  int BlockThreads,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename SegmentOpT,
  typename ElementTransformOpT,
  typename KernelSource =
    DeviceHierarchicalTransformKernelSource<BlockThreads, InputIteratorT, OutputIteratorT, SegmentOpT, ElementTransformOpT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHierarchicalTransform
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");

  InputIteratorT d_in;
  OutputIteratorT d_out;

  ::cuda::std::int64_t num_segments;
  int segment_size;

  SegmentOpT segment_op;
  ElementTransformOpT element_transform_op;

  cudaStream_t stream;
  int ptx_version;

  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchHierarchicalTransform(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    SegmentOpT segment_op,
    ElementTransformOpT element_transform_op,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_in(::cuda::std::move(d_in))
      , d_out(::cuda::std::move(d_out))
      , num_segments(num_segments)
      , segment_size(segment_size)
      , segment_op(::cuda::std::move(segment_op))
      , element_transform_op(::cuda::std::move(element_transform_op))
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeKernel(DeviceHierarchicalTransformKernelT hierarchical_transform_kernel)
  {
    if (num_segments == 0 || segment_size == 0)
    {
      return cudaSuccess;
    }

    int max_grid_dim_x = 0;
    if (const auto error = CubDebug(launcher_factory.MaxGridDimX(max_grid_dim_x)))
    {
      return error;
    }

    const auto num_segments_per_invocation =
      (::cuda::std::min) (static_cast<::cuda::std::int64_t>(max_grid_dim_x),
                          static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<int>::max()));

    for (::cuda::std::int64_t segment_offset = 0; segment_offset < num_segments;
         segment_offset += num_segments_per_invocation)
    {
      const auto num_current_segments = (::cuda::std::min) (num_segments_per_invocation, num_segments - segment_offset);
      const auto item_offset          = segment_offset * static_cast<::cuda::std::int64_t>(segment_size);

      launcher_factory(static_cast<int>(num_current_segments), BlockThreads, 0, stream)
        .doit(hierarchical_transform_kernel,
              d_in + item_offset,
              d_out + item_offset,
              static_cast<int>(num_current_segments),
              segment_size,
              segment_op,
              element_transform_op);

      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }

      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }
    }

    return cudaSuccess;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return InvokeKernel(kernel_source.HierarchicalTransformKernel());
  }
};

template <
  int BlockThreads = 256,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename SegmentOpT,
  typename ElementTransformOpT,
  typename KernelSource =
    DeviceHierarchicalTransformKernelSource<BlockThreads, InputIteratorT, OutputIteratorT, SegmentOpT, ElementTransformOpT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  int segment_size,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op,
  cudaStream_t stream,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  int ptx_version = 0;
  if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
  {
    return error;
  }

  return DispatchHierarchicalTransform<BlockThreads,
                                       InputIteratorT,
                                       OutputIteratorT,
                                       SegmentOpT,
                                       ElementTransformOpT,
                                       KernelSource,
                                       KernelLauncherFactory>{
    ::cuda::std::move(d_in),
    ::cuda::std::move(d_out),
    num_segments,
    segment_size,
    ::cuda::std::move(segment_op),
    ::cuda::std::move(element_transform_op),
    stream,
    ptx_version,
    kernel_source,
    launcher_factory}
    .Invoke();
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
