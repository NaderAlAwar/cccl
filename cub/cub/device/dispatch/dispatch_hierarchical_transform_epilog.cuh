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
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
struct NoopDeviceEpilogOp
{
  template <typename BlockGroupT, typename ResultsT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(BlockGroupT, ResultsT const&) const
  {}
};

template <typename T>
struct is_cuda_std_tuple : ::cuda::std::false_type
{};

template <typename... Ts>
struct is_cuda_std_tuple<::cuda::std::tuple<Ts...>> : ::cuda::std::true_type
{};

template <typename InputIteratorT>
using transform_epilog_input_tuple_t =
  ::cuda::std::conditional_t<is_cuda_std_tuple<::cuda::std::decay_t<InputIteratorT>>::value,
                             ::cuda::std::decay_t<InputIteratorT>,
                             ::cuda::std::tuple<::cuda::std::decay_t<InputIteratorT>>>;

template <typename InputIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto make_transform_epilog_input_tuple(InputIteratorT&& d_in)
{
  if constexpr (is_cuda_std_tuple<::cuda::std::decay_t<InputIteratorT>>::value)
  {
    return ::cuda::std::forward<InputIteratorT>(d_in);
  }
  else
  {
    return ::cuda::std::make_tuple(::cuda::std::forward<InputIteratorT>(d_in));
  }
}

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT>
struct DeviceHierarchicalTransformEpilogKernelSource;

template <int BlockThreads,
          int ItemsPerThread,
          typename... InputIteratorTs,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT>
struct DeviceHierarchicalTransformEpilogKernelSource<
  BlockThreads,
  ItemsPerThread,
  ::cuda::std::tuple<InputIteratorTs...>,
  OutputIteratorT,
  TransformOpT,
  DeviceEpilogOpT>
{
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformEpilogKernel,
    DeviceHierarchicalTransformEpilogKernel<BlockThreads,
                                            ItemsPerThread,
                                            OutputIteratorT,
                                            TransformOpT,
                                            DeviceEpilogOpT,
                                            InputIteratorTs...>)
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

  template <typename DeviceHierarchicalTransformEpilogKernelT, ::cuda::std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokeKernel(
    DeviceHierarchicalTransformEpilogKernelT hierarchical_transform_epilog_kernel, ::cuda::std::index_sequence<Is...>)
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
      .doit(hierarchical_transform_epilog_kernel,
            ::cuda::std::get<Is>(d_in)...,
            d_out,
            num_segments,
            segment_size,
            transform_op,
            device_epilog_op);

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
    return InvokeKernel(kernel_source.HierarchicalTransformEpilogKernel(),
                        ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<InputIteratorT>>{});
  }
};

template <int BlockThreads   = 256,
          int ItemsPerThread = 1,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename InputIteratorTupleT = transform_epilog_input_tuple_t<InputIteratorT>,
          typename KernelSource        = DeviceHierarchicalTransformEpilogKernelSource<
                   BlockThreads,
                   ItemsPerThread,
                   InputIteratorTupleT,
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
    InputIteratorTupleT,
    OutputIteratorT,
    TransformOpT,
    NoopDeviceEpilogOp,
    InputIteratorTupleT,
    KernelSource,
    KernelLauncherFactory>(
    make_transform_epilog_input_tuple(::cuda::std::move(d_in)),
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
          typename InputIteratorTupleT = transform_epilog_input_tuple_t<InputIteratorT>,
          typename KernelSource        = DeviceHierarchicalTransformEpilogKernelSource<
                   BlockThreads,
                   ItemsPerThread,
                   InputIteratorTupleT,
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
           InputIteratorTupleT,
           OutputIteratorT,
           TransformOpT,
           DeviceEpilogOpT,
           KernelSource,
           KernelLauncherFactory>{
    make_transform_epilog_input_tuple(::cuda::std::move(d_in)),
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
