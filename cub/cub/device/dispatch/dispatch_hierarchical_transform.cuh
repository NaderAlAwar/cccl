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
constexpr int transform_prolog_cluster_smem_threshold      = 64 * 1024;
constexpr int transform_prolog_block_large_block_threads   = 512;
constexpr int transform_prolog_default_items_per_thread    = 4;
constexpr int transform_prolog_max_portable_cluster_size   = 8;
constexpr int transform_prolog_cluster_large_block_threads = 512;
constexpr int transform_prolog_block_policy_cache_size     = 8;
constexpr int transform_prolog_cluster_policy_cache_size   = 8;

template <int BlockThreads, typename InputIteratorT, typename SegmentOpT>
inline constexpr bool hierarchical_transform_block_segment_op_v =
  transform_prolog_segment_range_selector<BlockThreads,
                                          ::cuda::experimental::this_block<decltype(::cuda::hierarchy(
                                            ::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()))>,
                                          SegmentOpT,
                                          cub::detail::it_value_t<InputIteratorT>>::valid;

template <int BlockThreads, typename InputIteratorT, typename SegmentOpT>
inline constexpr bool hierarchical_transform_cluster_segment_op_v = transform_prolog_segment_range_selector<
  BlockThreads,
  ::cuda::experimental::this_cluster<decltype(::cuda::hierarchy(
    ::cuda::grid_dims(dim3{}), ::cuda::cluster_dims(dim3{}), ::cuda::block_dims<BlockThreads>()))>,
  SegmentOpT,
  cub::detail::it_value_t<InputIteratorT>>::valid;

template <int BlockThreads,
          typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
struct DeviceHierarchicalTransformKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformKernel,
    DeviceHierarchicalTransformKernel<BlockThreads,
                                      InputIteratorT,
                                      DirectInputIteratorT,
                                      OutputIteratorT,
                                      SegmentOpT,
                                      ElementTransformOpT>)
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformClusterKernel,
    DeviceHierarchicalTransformClusterKernel<BlockThreads,
                                             InputIteratorT,
                                             DirectInputIteratorT,
                                             OutputIteratorT,
                                             SegmentOpT,
                                             ElementTransformOpT>)
};

template <int BlockThreads,
          typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<BlockThreads,
                                                                                   InputIteratorT,
                                                                                   DirectInputIteratorT,
                                                                                   OutputIteratorT,
                                                                                   SegmentOpT,
                                                                                   ElementTransformOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHierarchicalTransform
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(hierarchical_transform_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input values to be trivially relocatable.");

  InputIteratorT d_in;
  DirectInputIteratorT d_direct;
  OutputIteratorT d_out;

  ::cuda::std::int64_t num_segments;
  int segment_size;
  int items_per_thread;

  SegmentOpT segment_op;
  ElementTransformOpT element_transform_op;

  cudaStream_t stream;
  int ptx_version;

  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  struct block_candidate_t
  {
    int block_threads{0};
    int active_blocks{0};
    int active_warps{0};
  };

  struct block_policy_t
  {
    int block_threads{0};
    int active_blocks{0};
    int active_warps{0};
  };

  struct block_policy_cache_t
  {
    bool valid{false};
    int device{-1};
    ::cuda::std::size_t requested_shared_bytes{0};
    block_policy_t policy{};
  };

  struct cluster_candidate_t
  {
    int block_threads{0};
    int active_clusters{0};
    int active_warps{0};
  };

  struct cluster_policy_t
  {
    int block_threads{0};
    int active_clusters{0};
    int active_warps{0};
  };

  struct cluster_policy_cache_t
  {
    bool valid{false};
    int device{-1};
    int cluster_size{0};
    int chunk_items{0};
    ::cuda::std::size_t cluster_shared_bytes{0};
    cluster_policy_t policy{};
  };

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchHierarchicalTransform(
    InputIteratorT d_in,
    DirectInputIteratorT d_direct,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    int items_per_thread,
    SegmentOpT segment_op,
    ElementTransformOpT element_transform_op,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_in(::cuda::std::move(d_in))
      , d_direct(::cuda::std::move(d_direct))
      , d_out(::cuda::std::move(d_out))
      , num_segments(num_segments)
      , segment_size(segment_size)
      , items_per_thread(items_per_thread)
      , segment_op(::cuda::std::move(segment_op))
      , element_transform_op(::cuda::std::move(element_transform_op))
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <int CandidateBlockThreads, typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokeKernel(
    DeviceHierarchicalTransformKernelT hierarchical_transform_kernel,
    ::cuda::std::size_t requested_shared_bytes,
    int max_grid_dim_x)
  {
    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    if (num_segments < 0 || segment_size <= 0)
    {
      return cudaErrorInvalidValue;
    }

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

    const auto num_segments_per_invocation =
      (::cuda::std::min) (static_cast<::cuda::std::int64_t>(max_grid_dim_x),
                          static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<int>::max()));

    for (::cuda::std::int64_t segment_offset = 0; segment_offset < num_segments;
         segment_offset += num_segments_per_invocation)
    {
      const auto num_current_segments = (::cuda::std::min) (num_segments_per_invocation, num_segments - segment_offset);
      const auto item_offset          = segment_offset * static_cast<::cuda::std::int64_t>(segment_size);

      launcher_factory(
        static_cast<int>(num_current_segments), CandidateBlockThreads, static_cast<int>(requested_shared_bytes), stream)
        .doit(hierarchical_transform_kernel,
              d_in + item_offset,
              d_direct,
              d_out + item_offset,
              segment_size,
              items_per_thread,
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

  template <int CandidateBlockThreads, typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t QueryBlockCandidate(
    DeviceHierarchicalTransformKernelT hierarchical_transform_kernel,
    ::cuda::std::size_t requested_shared_bytes,
    block_candidate_t& candidate)
  {
    candidate = block_candidate_t{CandidateBlockThreads, 0, 0};

    int max_dynamic_smem_size = 0;
    if (const auto error =
          CubDebug(launcher_factory.max_dynamic_smem_size_for(max_dynamic_smem_size, hierarchical_transform_kernel)))
    {
      return error;
    }

    if (requested_shared_bytes > static_cast<::cuda::std::size_t>(max_dynamic_smem_size))
    {
      return cudaErrorInvalidValue;
    }

    NV_IF_TARGET(NV_IS_HOST,
                 (if (const auto error = CubDebug(launcher_factory.set_max_dynamic_smem_size_for(
                        hierarchical_transform_kernel, static_cast<int>(requested_shared_bytes)))) { return error; }),
                 (return cudaErrorNotSupported;))

    int active_blocks = 0;
    if (const auto error = CubDebug(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          hierarchical_transform_kernel,
          CandidateBlockThreads,
          static_cast<::cuda::std::size_t>(requested_shared_bytes))))
    {
      return error;
    }

    if (active_blocks <= 0)
    {
      return cudaErrorInvalidValue;
    }

    candidate.active_blocks = active_blocks;
    candidate.active_warps  = active_blocks * (CandidateBlockThreads / cub::detail::warp_threads);
    return cudaSuccess;
  }

  template <typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SelectBlockPolicy(
    DeviceHierarchicalTransformKernelT hierarchical_transform_kernel,
    int current_device,
    ::cuda::std::size_t requested_shared_bytes,
    block_policy_t& policy)
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      ({
        static block_policy_cache_t cached_policies[transform_prolog_block_policy_cache_size]{};
        static int next_cached_policy = 0;

        for (const auto& cached_policy : cached_policies)
        {
          if (cached_policy.valid && cached_policy.device == current_device
              && cached_policy.requested_shared_bytes == requested_shared_bytes)
          {
            policy = cached_policy.policy;
            return cudaSuccess;
          }
        }

        block_candidate_t selected_candidate{};
        if (const auto error = QueryBlockCandidate<BlockThreads>(
              hierarchical_transform_kernel, requested_shared_bytes, selected_candidate))
        {
          return error;
        }

        if constexpr (BlockThreads != transform_prolog_block_large_block_threads
                      && hierarchical_transform_block_segment_op_v<transform_prolog_block_large_block_threads,
                                                                   InputIteratorT,
                                                                   SegmentOpT>)
        {
          if (segment_size >= transform_prolog_block_large_block_threads)
          {
            using large_kernel_source_t = DeviceHierarchicalTransformKernelSource<
              transform_prolog_block_large_block_threads,
              InputIteratorT,
              DirectInputIteratorT,
              OutputIteratorT,
              SegmentOpT,
              ElementTransformOpT>;

            block_candidate_t large_candidate{};
            const auto large_error = QueryBlockCandidate<transform_prolog_block_large_block_threads>(
              large_kernel_source_t{}.HierarchicalTransformKernel(), requested_shared_bytes, large_candidate);

            if (large_error == cudaSuccess && large_candidate.active_warps > selected_candidate.active_warps)
            {
              selected_candidate = large_candidate;
            }
          }
        }

        policy = block_policy_t{
          selected_candidate.block_threads, selected_candidate.active_blocks, selected_candidate.active_warps};

        cached_policies[next_cached_policy] =
          block_policy_cache_t{true, current_device, requested_shared_bytes, policy};
        next_cached_policy = (next_cached_policy + 1) % transform_prolog_block_policy_cache_size;
        return cudaSuccess;
      }),
      (return cudaErrorNotSupported;))
  }

  template <int CandidateBlockThreads, typename DeviceHierarchicalTransformClusterKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokeClusterKernel(
    DeviceHierarchicalTransformClusterKernelT hierarchical_transform_cluster_kernel,
    int cluster_size,
    int chunk_items,
    ::cuda::std::size_t cluster_shared_bytes,
    int max_grid_dim_x)
  {
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
      const auto num_current_segments = (::cuda::std::min) (max_segments_per_invocation, num_segments - segment_offset);
      const auto item_offset          = segment_offset * static_cast<::cuda::std::int64_t>(segment_size);

      cudaLaunchConfig_t launch_config{};
      launch_config.gridDim          = dim3(static_cast<unsigned int>(num_current_segments * cluster_size), 1, 1);
      launch_config.blockDim         = dim3(CandidateBlockThreads, 1, 1);
      launch_config.dynamicSmemBytes = cluster_shared_bytes;
      launch_config.stream           = stream;
      launch_config.attrs            = &launch_attribute;
      launch_config.numAttrs         = 1;

      if (const auto error = CubDebug(cudaLaunchKernelEx(
            &launch_config,
            hierarchical_transform_cluster_kernel,
            d_in + item_offset,
            d_direct,
            d_out + item_offset,
            segment_size,
            items_per_thread,
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

  template <int CandidateBlockThreads, typename DeviceHierarchicalTransformClusterKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t QueryClusterCandidate(
    DeviceHierarchicalTransformClusterKernelT hierarchical_transform_cluster_kernel,
    int cluster_size,
    ::cuda::std::size_t cluster_shared_bytes,
    cluster_candidate_t& candidate)
  {
    candidate = cluster_candidate_t{CandidateBlockThreads, 0, 0};

    int max_dynamic_smem_size = 0;
    if (const auto error = CubDebug(
          launcher_factory.max_dynamic_smem_size_for(max_dynamic_smem_size, hierarchical_transform_cluster_kernel)))
    {
      return error;
    }

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
    occupancy_config.blockDim         = dim3(CandidateBlockThreads, 1, 1);
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

    candidate.active_clusters = active_clusters;
    candidate.active_warps    = active_clusters * cluster_size * (CandidateBlockThreads / cub::detail::warp_threads);
    return cudaSuccess;
  }

  template <typename DeviceHierarchicalTransformClusterKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SelectClusterPolicy(
    DeviceHierarchicalTransformClusterKernelT hierarchical_transform_cluster_kernel,
    int current_device,
    int cluster_size,
    int chunk_items,
    ::cuda::std::size_t cluster_shared_bytes,
    cluster_policy_t& policy)
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      ({
        static cluster_policy_cache_t cached_policies[transform_prolog_cluster_policy_cache_size]{};
        static int next_cached_policy = 0;

        for (const auto& cached_policy : cached_policies)
        {
          if (cached_policy.valid && cached_policy.device == current_device
              && cached_policy.cluster_size == cluster_size && cached_policy.chunk_items == chunk_items
              && cached_policy.cluster_shared_bytes == cluster_shared_bytes)
          {
            policy = cached_policy.policy;
            return cudaSuccess;
          }
        }

        cluster_candidate_t selected_candidate{};
        if (const auto error = QueryClusterCandidate<BlockThreads>(
              hierarchical_transform_cluster_kernel, cluster_size, cluster_shared_bytes, selected_candidate))
        {
          return error;
        }

        if constexpr (BlockThreads != transform_prolog_cluster_large_block_threads
                      && hierarchical_transform_cluster_segment_op_v<transform_prolog_cluster_large_block_threads,
                                                                     InputIteratorT,
                                                                     SegmentOpT>)
        {
          if (chunk_items >= transform_prolog_cluster_large_block_threads)
          {
            using large_kernel_source_t = DeviceHierarchicalTransformKernelSource<
              transform_prolog_cluster_large_block_threads,
              InputIteratorT,
              DirectInputIteratorT,
              OutputIteratorT,
              SegmentOpT,
              ElementTransformOpT>;

            cluster_candidate_t large_candidate{};
            const auto large_error = QueryClusterCandidate<transform_prolog_cluster_large_block_threads>(
              large_kernel_source_t{}.HierarchicalTransformClusterKernel(),
              cluster_size,
              cluster_shared_bytes,
              large_candidate);

            if (large_error == cudaSuccess && large_candidate.active_warps > selected_candidate.active_warps)
            {
              selected_candidate = large_candidate;
            }
          }
        }

        policy = cluster_policy_t{
          selected_candidate.block_threads, selected_candidate.active_clusters, selected_candidate.active_warps};

        cached_policies[next_cached_policy] =
          cluster_policy_cache_t{true, current_device, cluster_size, chunk_items, cluster_shared_bytes, policy};
        next_cached_policy = (next_cached_policy + 1) % transform_prolog_cluster_policy_cache_size;
        return cudaSuccess;
      }),
      (return cudaErrorNotSupported;))
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

      const int chunk_items         = ::cuda::ceil_div(segment_size, cluster_size);
      const int bulk_copy_alignment = transform_prolog_bulk_copy_alignment_for_ptx(ptx_version);
      const int shared_buffer_alignment =
        transform_prolog_load_to_shared_buffer_alignment<value_t>(bulk_copy_alignment);
      const int alignment_padding = shared_buffer_alignment > 16 ? (shared_buffer_alignment - 16) : 0;
      const auto cluster_shared_bytes =
        static_cast<::cuda::std::size_t>(
          transform_prolog_load_to_shared_buffer_size<value_t>(chunk_items, bulk_copy_alignment))
        + alignment_padding;

      if (cluster_shared_bytes > static_cast<::cuda::std::size_t>(max_dynamic_smem_size))
      {
        return cudaErrorInvalidValue;
      }

      cluster_policy_t cluster_policy{};
      if (const auto error = SelectClusterPolicy(
            hierarchical_transform_cluster_kernel,
            current_device,
            cluster_size,
            chunk_items,
            cluster_shared_bytes,
            cluster_policy))
      {
        return error;
      }

      if constexpr (BlockThreads != transform_prolog_cluster_large_block_threads
                    && hierarchical_transform_cluster_segment_op_v<transform_prolog_cluster_large_block_threads,
                                                                   InputIteratorT,
                                                                   SegmentOpT>)
      {
        if (cluster_policy.block_threads == transform_prolog_cluster_large_block_threads)
        {
          using large_kernel_source_t = DeviceHierarchicalTransformKernelSource<
            transform_prolog_cluster_large_block_threads,
            InputIteratorT,
            DirectInputIteratorT,
            OutputIteratorT,
            SegmentOpT,
            ElementTransformOpT>;

          return InvokeClusterKernel<transform_prolog_cluster_large_block_threads>(
            large_kernel_source_t{}.HierarchicalTransformClusterKernel(),
            cluster_size,
            chunk_items,
            cluster_shared_bytes,
            max_grid_dim_x);
        }
      }

      return InvokeClusterKernel<BlockThreads>(
        hierarchical_transform_cluster_kernel, cluster_size, chunk_items, cluster_shared_bytes, max_grid_dim_x);
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

    if (items_per_thread <= 0)
    {
      return cudaErrorInvalidValue;
    }

    using value_t = cub::detail::it_value_t<InputIteratorT>;

    const int bulk_copy_alignment     = transform_prolog_bulk_copy_alignment_for_ptx(ptx_version);
    const int shared_buffer_alignment = transform_prolog_load_to_shared_buffer_alignment<value_t>(bulk_copy_alignment);
    const int alignment_padding       = shared_buffer_alignment > 16 ? (shared_buffer_alignment - 16) : 0;
    const auto requested_shared_bytes =
      static_cast<::cuda::std::size_t>(
        transform_prolog_load_to_shared_buffer_size<value_t>(segment_size, bulk_copy_alignment))
      + alignment_padding;

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

    int current_device = 0;
    if (const auto error = CubDebug(cudaGetDevice(&current_device)))
    {
      return error;
    }

    block_policy_t block_policy{};
    if (const auto error =
          SelectBlockPolicy(hierarchical_transform_kernel, current_device, requested_shared_bytes, block_policy))
    {
      return error;
    }

    if constexpr (BlockThreads != transform_prolog_block_large_block_threads
                  && hierarchical_transform_block_segment_op_v<transform_prolog_block_large_block_threads,
                                                               InputIteratorT,
                                                               SegmentOpT>)
    {
      if (block_policy.block_threads == transform_prolog_block_large_block_threads)
      {
        using large_kernel_source_t = DeviceHierarchicalTransformKernelSource<
          transform_prolog_block_large_block_threads,
          InputIteratorT,
          DirectInputIteratorT,
          OutputIteratorT,
          SegmentOpT,
          ElementTransformOpT>;

        return InvokeKernel<transform_prolog_block_large_block_threads>(
          large_kernel_source_t{}.HierarchicalTransformKernel(), requested_shared_bytes, max_grid_dim_x);
      }
    }

    return InvokeKernel<BlockThreads>(hierarchical_transform_kernel, requested_shared_bytes, max_grid_dim_x);
  }
};

template <int BlockThreads = 256,
          typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<BlockThreads,
                                                                                   InputIteratorT,
                                                                                   DirectInputIteratorT,
                                                                                   OutputIteratorT,
                                                                                   SegmentOpT,
                                                                                   ElementTransformOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  InputIteratorT d_in,
  DirectInputIteratorT d_direct,
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
                                       DirectInputIteratorT,
                                       OutputIteratorT,
                                       SegmentOpT,
                                       ElementTransformOpT,
                                       KernelSource,
                                       KernelLauncherFactory>{
    ::cuda::std::move(d_in),
    ::cuda::std::move(d_direct),
    ::cuda::std::move(d_out),
    num_segments,
    segment_size,
    transform_prolog_default_items_per_thread,
    ::cuda::std::move(segment_op),
    ::cuda::std::move(element_transform_op),
    stream,
    ptx_version,
    kernel_source,
    launcher_factory}
    .Invoke();
}

template <int BlockThreads = 256,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<BlockThreads,
                                                                                   InputIteratorT,
                                                                                   transform_prolog_no_direct_input,
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
                                       InputIteratorT,
                                       transform_prolog_no_direct_input,
                                       OutputIteratorT,
                                       SegmentOpT,
                                       ElementTransformOpT,
                                       KernelSource,
                                       KernelLauncherFactory>{
    ::cuda::std::move(d_in),
    transform_prolog_no_direct_input{},
    ::cuda::std::move(d_out),
    num_segments,
    segment_size,
    transform_prolog_default_items_per_thread,
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
