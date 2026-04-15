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
#include <cub/device/dispatch/kernels/kernel_hierarchical_segmented_reduce.cuh>
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
          typename SegmentOutputIteratorT,
          typename ReductionOpT,
          typename InitT,
          typename DeviceEpilogOpT>
struct DeviceHierarchicalSegmentedReduceKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalSegmentedReduceKernel,
    DeviceHierarchicalSegmentedReduceKernel<BlockThreads,
                                            InputIteratorT,
                                            SegmentOutputIteratorT,
                                            ReductionOpT,
                                            InitT,
                                            DeviceEpilogOpT>)
};

template <int BlockThreads,
          typename InputIteratorT,
          typename SegmentOutputIteratorT,
          typename ReductionOpT,
          typename InitT,
          typename DeviceEpilogOpT,
          typename KernelSource = DeviceHierarchicalSegmentedReduceKernelSource<
            BlockThreads,
            InputIteratorT,
            SegmentOutputIteratorT,
            ReductionOpT,
            InitT,
            DeviceEpilogOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHierarchicalSegmentedReduce
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");

  InputIteratorT d_in;
  SegmentOutputIteratorT d_segment_out;

  ::cuda::std::int64_t num_segments;
  ::cuda::std::int64_t num_items;
  int segment_size;

  ReductionOpT reduction_op;
  InitT init;
  DeviceEpilogOpT device_epilog_op;

  cudaStream_t stream;
  int ptx_version;

  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchHierarchicalSegmentedReduce(
    InputIteratorT d_in,
    SegmentOutputIteratorT d_segment_out,
    ::cuda::std::int64_t num_segments,
    ::cuda::std::int64_t num_items,
    int segment_size,
    ReductionOpT reduction_op,
    InitT init,
    DeviceEpilogOpT device_epilog_op,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_in(::cuda::std::move(d_in))
      , d_segment_out(::cuda::std::move(d_segment_out))
      , num_segments(num_segments)
      , num_items(num_items)
      , segment_size(segment_size)
      , reduction_op(::cuda::std::move(reduction_op))
      , init(::cuda::std::move(init))
      , device_epilog_op(::cuda::std::move(device_epilog_op))
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <typename DeviceHierarchicalSegmentedReduceKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeKernel(DeviceHierarchicalSegmentedReduceKernelT hierarchical_segmented_reduce_kernel)
  {
    if (num_segments == 0 || num_items == 0 || segment_size == 0)
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
        .doit(hierarchical_segmented_reduce_kernel,
              d_in + item_offset,
              d_segment_out + segment_offset,
              static_cast<int>(num_current_segments),
              num_items - item_offset,
              segment_size,
              reduction_op,
              init,
              device_epilog_op);

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
    return InvokeKernel(kernel_source.HierarchicalSegmentedReduceKernel());
  }
};

template <int BlockThreads = 256,
          typename InputIteratorT,
          typename SegmentOutputIteratorT,
          typename ReductionOpT,
          typename InitT,
          typename DeviceEpilogOpT,
          typename KernelSource = DeviceHierarchicalSegmentedReduceKernelSource<
            BlockThreads,
            InputIteratorT,
            SegmentOutputIteratorT,
            ReductionOpT,
            InitT,
            DeviceEpilogOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_segmented_reduce(
  InputIteratorT d_in,
  SegmentOutputIteratorT d_segment_out,
  ::cuda::std::int64_t num_segments,
  ::cuda::std::int64_t num_items,
  int segment_size,
  ReductionOpT reduction_op,
  InitT init,
  DeviceEpilogOpT device_epilog_op,
  cudaStream_t stream,
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  int ptx_version = 0;
  if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
  {
    return error;
  }

  return DispatchHierarchicalSegmentedReduce<
           BlockThreads,
           InputIteratorT,
           SegmentOutputIteratorT,
           ReductionOpT,
           InitT,
           DeviceEpilogOpT,
           KernelSource,
           KernelLauncherFactory>{
    ::cuda::std::move(d_in),
    ::cuda::std::move(d_segment_out),
    num_segments,
    num_items,
    segment_size,
    ::cuda::std::move(reduction_op),
    ::cuda::std::move(init),
    ::cuda::std::move(device_epilog_op),
    stream,
    ptx_version,
    kernel_source,
    launcher_factory}
    .Invoke();
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
