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
#include <cub/device/dispatch/kernels/kernel_hierarchical_transform_epilog.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
struct NoopDeviceEpilogOp
{
  template <typename BlockGroupT, typename ResultsT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(BlockGroupT, ResultsT const&) const
  {}
};

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT>
struct DeviceHierarchicalTransformEpilogKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformEpilogKernel,
    DeviceHierarchicalTransformEpilogKernel<BlockThreads,
                                            ItemsPerThread,
                                            InputIteratorT,
                                            OutputIteratorT,
                                            TransformOpT,
                                            DeviceEpilogOpT>)
};

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT,
          typename KernelSource = DeviceHierarchicalTransformEpilogKernelSource<
            BlockThreads,
            ItemsPerThread,
            InputIteratorT,
            OutputIteratorT,
            TransformOpT,
            DeviceEpilogOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHierarchicalTransformEpilog
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");

  InputIteratorT d_in;
  OutputIteratorT d_out;

  ::cuda::std::int64_t num_segments;
  int segment_size;

  TransformOpT transform_op;
  DeviceEpilogOpT device_epilog_op;

  cudaStream_t stream;

  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchHierarchicalTransformEpilog(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    TransformOpT transform_op,
    DeviceEpilogOpT device_epilog_op,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_in(::cuda::std::move(d_in))
      , d_out(::cuda::std::move(d_out))
      , num_segments(num_segments)
      , segment_size(segment_size)
      , transform_op(::cuda::std::move(transform_op))
      , device_epilog_op(::cuda::std::move(device_epilog_op))
      , stream(stream)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <typename DeviceHierarchicalTransformEpilogKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeKernel(DeviceHierarchicalTransformEpilogKernelT hierarchical_transform_epilog_kernel)
  {
    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    constexpr int warp_threads    = 32;
    constexpr int warps_per_block = BlockThreads / warp_threads;

    if (segment_size <= 0 || segment_size > warp_threads)
    {
      return cudaErrorInvalidValue;
    }

    int max_grid_dim_x = 0;
    if (const auto error = CubDebug(launcher_factory.MaxGridDimX(max_grid_dim_x)))
    {
      return error;
    }

    const auto required_blocks = (num_segments + warps_per_block - 1) / warps_per_block;
    const auto grid_size = (::cuda::std::min) (required_blocks, static_cast<::cuda::std::int64_t>(max_grid_dim_x));

    launcher_factory(static_cast<int>(grid_size), BlockThreads, 0, stream)
      .doit(
        hierarchical_transform_epilog_kernel, d_in, d_out, num_segments, segment_size, transform_op, device_epilog_op);

    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    return cudaSuccess;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return InvokeKernel(kernel_source.HierarchicalTransformEpilogKernel());
  }
};

template <int BlockThreads   = 256,
          int ItemsPerThread = 1,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename KernelSource = DeviceHierarchicalTransformEpilogKernelSource<
            BlockThreads,
            ItemsPerThread,
            InputIteratorT,
            OutputIteratorT,
            TransformOpT,
            NoopDeviceEpilogOp>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_transform_epilog(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  int segment_size,
  TransformOpT transform_op,
  cudaStream_t stream,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  return dispatch_transform_epilog<
    BlockThreads,
    ItemsPerThread,
    InputIteratorT,
    OutputIteratorT,
    TransformOpT,
    NoopDeviceEpilogOp,
    KernelSource,
    KernelLauncherFactory>(
    ::cuda::std::move(d_in),
    ::cuda::std::move(d_out),
    num_segments,
    segment_size,
    ::cuda::std::move(transform_op),
    NoopDeviceEpilogOp{},
    stream,
    kernel_source,
    launcher_factory);
}

template <int BlockThreads   = 256,
          int ItemsPerThread = 1,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT,
          typename KernelSource = DeviceHierarchicalTransformEpilogKernelSource<
            BlockThreads,
            ItemsPerThread,
            InputIteratorT,
            OutputIteratorT,
            TransformOpT,
            DeviceEpilogOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_transform_epilog(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  int segment_size,
  TransformOpT transform_op,
  DeviceEpilogOpT device_epilog_op,
  cudaStream_t stream,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  return DispatchHierarchicalTransformEpilog<
           BlockThreads,
           ItemsPerThread,
           InputIteratorT,
           OutputIteratorT,
           TransformOpT,
           DeviceEpilogOpT,
           KernelSource,
           KernelLauncherFactory>{
    ::cuda::std::move(d_in),
    ::cuda::std::move(d_out),
    num_segments,
    segment_size,
    ::cuda::std::move(transform_op),
    ::cuda::std::move(device_epilog_op),
    stream,
    kernel_source,
    launcher_factory}
    .Invoke();
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
