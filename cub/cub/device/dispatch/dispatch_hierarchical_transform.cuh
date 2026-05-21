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
#include <cub/device/dispatch/tuning/tuning_transform.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
struct DeviceHierarchicalTransformKernelSource
{
  template <int BlockThreads, int BulkCopyAlignment, bool VectorizeDirectAndOutput>
  _CCCL_HIDE_FROM_ABI
  CUB_RUNTIME_FUNCTION static constexpr decltype(&DeviceHierarchicalTransformKernel<
                                                 BlockThreads,
                                                 BulkCopyAlignment,
                                                 VectorizeDirectAndOutput,
                                                 InputIteratorT,
                                                 DirectInputIteratorT,
                                                 OutputIteratorT,
                                                 SegmentOpT,
                                                 ElementTransformOpT>)
  HierarchicalTransformKernel()
  {
    return &DeviceHierarchicalTransformKernel<
      BlockThreads,
      BulkCopyAlignment,
      VectorizeDirectAndOutput,
      InputIteratorT,
      DirectInputIteratorT,
      OutputIteratorT,
      SegmentOpT,
      ElementTransformOpT>;
  }
};

template <typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<InputIteratorT,
                                                                                   DirectInputIteratorT,
                                                                                   OutputIteratorT,
                                                                                   SegmentOpT,
                                                                                   ElementTransformOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchHierarchicalTransform
{
  using input_value_t = cub::detail::it_value_t<InputIteratorT>;

  static_assert(THRUST_NS_QUALIFIER::is_trivially_relocatable_v<::cuda::std::remove_cv_t<input_value_t>>,
                "TransformProlog requires input values to be trivially relocatable.");

  // TransformProlog stages one full segment in shared memory, so occupancy depends on runtime segment size rather than
  // only a static policy. Try a larger block when shared memory limits the number of resident CTAs.
  static constexpr int default_block_threads   = 256;
  static constexpr int large_block_threads     = 512;
  static constexpr int block_policy_cache_size = 8;

  InputIteratorT d_in;
  DirectInputIteratorT d_direct;
  OutputIteratorT d_out;

  ::cuda::std::int64_t num_segments;
  int segment_size;

  SegmentOpT segment_op;
  ElementTransformOpT element_transform_op;

  cudaStream_t stream;
  ::cuda::arch_id arch_id;

  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchHierarchicalTransform(
    InputIteratorT d_in,
    DirectInputIteratorT d_direct,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    SegmentOpT segment_op,
    ElementTransformOpT element_transform_op,
    cudaStream_t stream,
    ::cuda::arch_id arch_id,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_in(::cuda::std::move(d_in))
      , d_direct(::cuda::std::move(d_direct))
      , d_out(::cuda::std::move(d_out))
      , num_segments(num_segments)
      , segment_size(segment_size)
      , segment_op(::cuda::std::move(segment_op))
      , element_transform_op(::cuda::std::move(element_transform_op))
      , stream(stream)
      , arch_id(arch_id)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <int CandidateBlockThreads, typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokeKernel(
    DeviceHierarchicalTransformKernelT hierarchical_transform_kernel,
    ::cuda::std::size_t requested_shared_bytes,
    int max_grid_dim_x)
  {
    if (num_segments < 0 || segment_size <= 0)
    {
      return cudaErrorInvalidValue;
    }

    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    if (max_grid_dim_x <= 0)
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

  struct block_policy_t
  {
    int block_threads{0};
    int active_warps{0};
  };

  template <int CandidateBlockThreads, typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t QueryBlockPolicy(
    DeviceHierarchicalTransformKernelT hierarchical_transform_kernel,
    ::cuda::std::size_t requested_shared_bytes,
    block_policy_t& policy)
  {
    policy = block_policy_t{CandidateBlockThreads, 0};

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

    int active_blocks = 0;
    if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
          active_blocks,
          hierarchical_transform_kernel,
          CandidateBlockThreads,
          static_cast<int>(requested_shared_bytes))))
    {
      return error;
    }

    if (active_blocks <= 0)
    {
      return cudaErrorInvalidValue;
    }

    policy.active_warps = active_blocks * (CandidateBlockThreads / cub::detail::warp_threads);
    return cudaSuccess;
  }

  struct block_policy_cache_t
  {
    bool valid{false};
    int device{-1};
    int segment_size{0};
    ::cuda::std::size_t requested_shared_bytes{0};
    bool vectorize_direct_and_output{false};
    block_policy_t policy{};
  };

  template <int BulkCopyAlignment, bool VectorizeDirectAndOutput, typename DeviceHierarchicalTransformKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SelectBlockPolicy(
    DeviceHierarchicalTransformKernelT default_hierarchical_transform_kernel,
    int current_device,
    ::cuda::std::size_t requested_shared_bytes,
    block_policy_t& policy)
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      ({
        static block_policy_cache_t cached_policies[block_policy_cache_size]{};
        static int next_cached_policy = 0;

        for (const auto& cached_policy : cached_policies)
        {
          if (cached_policy.valid && cached_policy.device == current_device
              && cached_policy.segment_size == segment_size
              && cached_policy.requested_shared_bytes == requested_shared_bytes
              && cached_policy.vectorize_direct_and_output == VectorizeDirectAndOutput)
          {
            policy = cached_policy.policy;
            return cudaSuccess;
          }
        }

        block_policy_t selected_policy{};
        if (const auto error = QueryBlockPolicy<default_block_threads>(
              default_hierarchical_transform_kernel, requested_shared_bytes, selected_policy))
        {
          return error;
        }

        if (segment_size >= large_block_threads)
        {
          block_policy_t large_policy{};
          const auto large_error = QueryBlockPolicy<large_block_threads>(
            kernel_source
              .template HierarchicalTransformKernel<large_block_threads, BulkCopyAlignment, VectorizeDirectAndOutput>(),
            requested_shared_bytes,
            large_policy);

          if (large_error == cudaSuccess && large_policy.active_warps > selected_policy.active_warps)
          {
            selected_policy = large_policy;
          }
        }

        policy                              = selected_policy;
        cached_policies[next_cached_policy] = block_policy_cache_t{
          true, current_device, segment_size, requested_shared_bytes, VectorizeDirectAndOutput, policy};
        next_cached_policy = (next_cached_policy + 1) % block_policy_cache_size;
        return cudaSuccess;
      }),
      (return cudaErrorNotSupported;))
  }

  template <int BulkCopyAlignment>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE bool CanVectorizeDirectAndOutput()
  {
    using direct_vector_load_t = transform_prolog_direct_vector_load<::cuda::std::remove_cv_t<DirectInputIteratorT>>;
    constexpr bool output_is_contiguous_f32_pointer =
      ::cuda::std::is_pointer_v<::cuda::std::remove_cv_t<OutputIteratorT>>
      && ::cuda::std::is_same_v<
        ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::remove_cv_t<OutputIteratorT>>>,
        float>;
    constexpr bool uses_vectorized_store_outputs =
      transform_prolog_shared_vector_aligned_v<BulkCopyAlignment>
      && ::cuda::std::is_same_v<::cuda::std::remove_cv_t<input_value_t>, float> && direct_vector_load_t::supported
      && output_is_contiguous_f32_pointer;

    if constexpr (uses_vectorized_store_outputs)
    {
      constexpr int vector_items = sizeof(transform_prolog_f32_vector_storage_t) / sizeof(float);
      if (segment_size % vector_items != 0)
      {
        return false;
      }

      if (!direct_vector_load_t::IsAligned(d_direct, 0))
      {
        return false;
      }

      constexpr auto output_alignment =
        static_cast<::cuda::std::uintptr_t>(alignof(transform_prolog_f32_vector_storage_t));
      const auto output_address = reinterpret_cast<::cuda::std::uintptr_t>(d_out);
      if (output_address % output_alignment != 0)
      {
        return false;
      }

      return true;
    }

    return false;
  }

  template <int BulkCopyAlignment, bool VectorizeDirectAndOutput>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeAligned()
  {
    if (num_segments < 0 || segment_size <= 0)
    {
      return cudaErrorInvalidValue;
    }

    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    int max_grid_dim_x = 0;
    if (const auto error = CubDebug(launcher_factory.MaxGridDimX(max_grid_dim_x)))
    {
      return error;
    }

    int current_device = 0;
    if (const auto error = CubDebug(cudaGetDevice(&current_device)))
    {
      return error;
    }

    auto hierarchical_transform_kernel =
      kernel_source
        .template HierarchicalTransformKernel<default_block_threads, BulkCopyAlignment, VectorizeDirectAndOutput>();

    const auto requested_shared_bytes = static_cast<::cuda::std::size_t>(
      cub::detail::LoadToSharedBufferSizeBytes<input_value_t, BulkCopyAlignment>(segment_size));

    block_policy_t block_policy{};
    if (const auto error = SelectBlockPolicy<BulkCopyAlignment, VectorizeDirectAndOutput>(
          hierarchical_transform_kernel, current_device, requested_shared_bytes, block_policy))
    {
      return error;
    }

    if (block_policy.block_threads == large_block_threads)
    {
      return InvokeKernel<large_block_threads>(
        kernel_source
          .template HierarchicalTransformKernel<large_block_threads, BulkCopyAlignment, VectorizeDirectAndOutput>(),
        requested_shared_bytes,
        max_grid_dim_x);
    }

    return InvokeKernel<default_block_threads>(hierarchical_transform_kernel, requested_shared_bytes, max_grid_dim_x);
  }

  template <int BulkCopyAlignment, typename MaybePointerT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static bool CanUseBulkCopyAlignment(MaybePointerT input, int items)
  {
    if constexpr (::cuda::std::is_pointer_v<::cuda::std::decay_t<MaybePointerT>>)
    {
      const auto address   = reinterpret_cast<::cuda::std::uintptr_t>(input);
      const auto num_bytes = static_cast<::cuda::std::uintptr_t>(items) * sizeof(input_value_t);
      const auto alignment = static_cast<::cuda::std::uintptr_t>(BulkCopyAlignment);
      const bool begin_ok  = (address % alignment) == 0;
      const bool range_ok  = (num_bytes % alignment) == 0;
      return begin_ok && range_ok;
    }
    else
    {
      return false;
    }
  }

  template <int PreferredBulkCopyAlignment>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokePreferredAlignment()
  {
    constexpr int bulk_copy_alignment =
      PreferredBulkCopyAlignment < static_cast<int>(alignof(input_value_t))
        ? static_cast<int>(alignof(input_value_t))
        : PreferredBulkCopyAlignment;

    if (CanUseBulkCopyAlignment<bulk_copy_alignment>(d_in, segment_size))
    {
      // The F32 direct/output vector path is all-or-nothing: if the direct input, output, or segment length cannot
      // support 16-byte vector accesses, keep the same staged-input alignment but use scalar direct/output accesses.
      if (CanVectorizeDirectAndOutput<bulk_copy_alignment>())
      {
        return InvokeAligned<bulk_copy_alignment, true>();
      }

      return InvokeAligned<bulk_copy_alignment, false>();
    }

    return InvokeAligned<static_cast<int>(alignof(input_value_t)), false>();
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    constexpr int sm90_bulk_copy_alignment  = transform::bulk_copy_alignment(::cuda::arch_id::sm_90);
    constexpr int sm100_bulk_copy_alignment = transform::bulk_copy_alignment(::cuda::arch_id::sm_100);

    // TODO: Move TransformProlog block size and bulk-copy alignment into an architecture-specific tuning policy.
    // BulkCopyAlignment is still a kernel template parameter, so dispatch maps the runtime policy result onto the
    // supported template instantiations here.
    if (transform::bulk_copy_alignment(arch_id) == sm90_bulk_copy_alignment)
    {
      return InvokePreferredAlignment<sm90_bulk_copy_alignment>();
    }

    return InvokePreferredAlignment<sm100_bulk_copy_alignment>();
  }
};

template <typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT,
          typename KernelSource          = DeviceHierarchicalTransformKernelSource<InputIteratorT,
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
  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

  return DispatchHierarchicalTransform<InputIteratorT,
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
    ::cuda::std::move(segment_op),
    ::cuda::std::move(element_transform_op),
    stream,
    arch_id,
    kernel_source,
    launcher_factory}
    .Invoke();
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
