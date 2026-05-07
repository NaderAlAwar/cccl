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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
constexpr int transform_prolog_cluster_smem_threshold    = 64 * 1024;
constexpr int transform_prolog_max_portable_cluster_size = 8;

template <int BlockThreads, typename InputIteratorT, typename SegmentOpT>
inline constexpr bool hierarchical_transform_cluster_segment_op_v = ::cuda::std::is_invocable_v<
  SegmentOpT,
  ::cuda::experimental::this_cluster<decltype(::cuda::hierarchy(
    ::cuda::grid_dims(dim3{}), ::cuda::cluster_dims(dim3{}), ::cuda::block_dims<BlockThreads>()))>,
  thread_segment_range<cub::detail::it_value_t<InputIteratorT>*>>;

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
struct DeviceHierarchicalTransformKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformKernel,
    DeviceHierarchicalTransformKernel<BlockThreads,
                                      ItemsPerThread,
                                      InputIteratorT,
                                      OutputIteratorT,
                                      SegmentOpT,
                                      ElementTransformOpT>)
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformClusterKernel,
    DeviceHierarchicalTransformClusterKernel<BlockThreads,
                                             ItemsPerThread,
                                             InputIteratorT,
                                             OutputIteratorT,
                                             SegmentOpT,
                                             ElementTransformOpT>)
};

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<BlockThreads,
                                                                                   ItemsPerThread,
                                                                                   InputIteratorT,
                                                                                   OutputIteratorT,
                                                                                   SegmentOpT,
                                                                                   ElementTransformOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHierarchicalTransform
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(ItemsPerThread > 0, "ItemsPerThread must be positive.");
  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");

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
    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    if (num_segments < 0 || segment_size <= 0)
    {
      return cudaErrorInvalidValue;
    }

    using value_t = cub::detail::it_value_t<InputIteratorT>;

    constexpr int shared_buffer_alignment = alignof(value_t);
    constexpr int alignment_padding       = shared_buffer_alignment > 16 ? (shared_buffer_alignment - 16) : 0;

    const auto requested_shared_bytes =
      static_cast<::cuda::std::size_t>(segment_size) * sizeof(value_t) + alignment_padding;

    int max_dynamic_smem_size = 0;
    if (const auto error =
          CubDebug(launcher_factory.max_dynamic_smem_size_for(max_dynamic_smem_size, hierarchical_transform_kernel)))
    {
      return error;
    }

    if (requested_shared_bytes <= static_cast<::cuda::std::size_t>(max_dynamic_smem_size))
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (if (const auto error = CubDebug(launcher_factory.set_max_dynamic_smem_size_for(
                          hierarchical_transform_kernel, static_cast<int>(requested_shared_bytes)))) { return error; }))
    }
    else
    {
      // TransformProlog currently requires staging the complete segment in dynamic shared memory.
      return cudaErrorInvalidValue;
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

      launcher_factory(
        static_cast<int>(num_current_segments), BlockThreads, static_cast<int>(requested_shared_bytes), stream)
        .doit(hierarchical_transform_kernel,
              d_in + item_offset,
              d_out + item_offset,
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

  template <typename DeviceHierarchicalTransformClusterKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokeClusterKernel(
    DeviceHierarchicalTransformClusterKernelT hierarchical_transform_cluster_kernel,
    ::cuda::std::size_t requested_shared_bytes,
    int max_grid_dim_x)
  {
    if constexpr (!hierarchical_transform_cluster_segment_op_v<BlockThreads, InputIteratorT, SegmentOpT>)
    {
      return cudaErrorInvalidValue;
    }
    else
    {
      if (ptx_version < 900)
      {
        return cudaErrorInvalidValue;
      }

      int current_device = 0;
      if (const auto error = CubDebug(cudaGetDevice(&current_device)))
      {
        return error;
      }

      int cluster_launch_supported = 0;
      if (const auto error =
            CubDebug(cudaDeviceGetAttribute(&cluster_launch_supported, cudaDevAttrClusterLaunch, current_device)))
      {
        return error;
      }

      if (!cluster_launch_supported)
      {
        return cudaErrorInvalidValue;
      }

      int max_dynamic_smem_size = 0;
      if (const auto error = CubDebug(
            launcher_factory.max_dynamic_smem_size_for(max_dynamic_smem_size, hierarchical_transform_cluster_kernel)))
      {
        return error;
      }

      const int target_smem_per_block =
        (::cuda::std::min) (max_dynamic_smem_size, transform_prolog_cluster_smem_threshold);
      if (target_smem_per_block <= 0)
      {
        return cudaErrorInvalidValue;
      }

      const int cluster_size = static_cast<int>(
        ::cuda::ceil_div(requested_shared_bytes, static_cast<::cuda::std::size_t>(target_smem_per_block)));
      if (cluster_size <= 1 || cluster_size > transform_prolog_max_portable_cluster_size)
      {
        return cudaErrorInvalidValue;
      }

      using value_t = cub::detail::it_value_t<InputIteratorT>;

      const int chunk_items                 = ::cuda::ceil_div(segment_size, cluster_size);
      constexpr int shared_buffer_alignment = alignof(value_t);
      constexpr int alignment_padding       = shared_buffer_alignment > 16 ? (shared_buffer_alignment - 16) : 0;
      const auto cluster_shared_bytes =
        static_cast<::cuda::std::size_t>(chunk_items) * sizeof(value_t) + alignment_padding;

      if (cluster_shared_bytes > static_cast<::cuda::std::size_t>(max_dynamic_smem_size))
      {
        return cudaErrorInvalidValue;
      }

      NV_IF_TARGET(
        NV_IS_HOST,
        (if (const auto error = CubDebug(launcher_factory.set_max_dynamic_smem_size_for(
               hierarchical_transform_cluster_kernel, static_cast<int>(cluster_shared_bytes)))) { return error; }),
        (return cudaErrorNotSupported;))

      cudaLaunchAttribute launch_attribute{};
      launch_attribute.id               = cudaLaunchAttributeClusterDimension;
      launch_attribute.val.clusterDim.x = static_cast<unsigned int>(cluster_size);
      launch_attribute.val.clusterDim.y = 1;
      launch_attribute.val.clusterDim.z = 1;

      cudaLaunchConfig_t occupancy_config{};
      occupancy_config.gridDim          = dim3(static_cast<unsigned int>(cluster_size), 1, 1);
      occupancy_config.blockDim         = dim3(BlockThreads, 1, 1);
      occupancy_config.dynamicSmemBytes = cluster_shared_bytes;
      occupancy_config.attrs            = &launch_attribute;
      occupancy_config.numAttrs         = 1;

      int active_clusters = 0;
      if (const auto error = CubDebug(
            cudaOccupancyMaxActiveClusters(&active_clusters, hierarchical_transform_cluster_kernel, &occupancy_config)))
      {
        return error;
      }

      if (active_clusters <= 0)
      {
        return cudaErrorInvalidValue;
      }

      const auto max_segments_per_invocation =
        (::cuda::std::min) (static_cast<::cuda::std::int64_t>(max_grid_dim_x / cluster_size),
                            static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<int>::max() / cluster_size));

      if (max_segments_per_invocation <= 0)
      {
        return cudaErrorInvalidValue;
      }

      for (::cuda::std::int64_t segment_offset = 0; segment_offset < num_segments;
           segment_offset += max_segments_per_invocation)
      {
        const auto num_current_segments =
          (::cuda::std::min) (max_segments_per_invocation, num_segments - segment_offset);
        const auto item_offset = segment_offset * static_cast<::cuda::std::int64_t>(segment_size);

        cudaLaunchConfig_t launch_config{};
        launch_config.gridDim          = dim3(static_cast<unsigned int>(num_current_segments * cluster_size), 1, 1);
        launch_config.blockDim         = dim3(BlockThreads, 1, 1);
        launch_config.dynamicSmemBytes = cluster_shared_bytes;
        launch_config.stream           = stream;
        launch_config.attrs            = &launch_attribute;
        launch_config.numAttrs         = 1;

        if (const auto error = CubDebug(cudaLaunchKernelEx(
              &launch_config,
              hierarchical_transform_cluster_kernel,
              d_in + item_offset,
              d_out + item_offset,
              segment_size,
              cluster_size,
              chunk_items,
              segment_op,
              element_transform_op)))
        {
          return error;
        }

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
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    auto hierarchical_transform_kernel = kernel_source.HierarchicalTransformKernel();

    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    if (num_segments < 0 || segment_size <= 0)
    {
      return cudaErrorInvalidValue;
    }

    using value_t = cub::detail::it_value_t<InputIteratorT>;

    constexpr int shared_buffer_alignment = alignof(value_t);
    constexpr int alignment_padding       = shared_buffer_alignment > 16 ? (shared_buffer_alignment - 16) : 0;
    const auto requested_shared_bytes =
      static_cast<::cuda::std::size_t>(segment_size) * sizeof(value_t) + alignment_padding;

    int max_grid_dim_x = 0;
    if (const auto error = CubDebug(launcher_factory.MaxGridDimX(max_grid_dim_x)))
    {
      return error;
    }

    if constexpr (hierarchical_transform_cluster_segment_op_v<BlockThreads, InputIteratorT, SegmentOpT>)
    {
      if (requested_shared_bytes > transform_prolog_cluster_smem_threshold)
      {
        const auto cluster_error = InvokeClusterKernel(
          kernel_source.HierarchicalTransformClusterKernel(), requested_shared_bytes, max_grid_dim_x);
        if (cluster_error == cudaSuccess)
        {
          return cudaSuccess;
        }
        if (cluster_error != cudaErrorInvalidValue && cluster_error != cudaErrorNotSupported)
        {
          return cluster_error;
        }
      }
    }

    return InvokeKernel(hierarchical_transform_kernel);
  }
};

template <int BlockThreads   = 256,
          int ItemsPerThread = 4,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<BlockThreads,
                                                                                   ItemsPerThread,
                                                                                   InputIteratorT,
                                                                                   OutputIteratorT,
                                                                                   SegmentOpT,
                                                                                   ElementTransformOpT>,
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
                                       ItemsPerThread,
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
