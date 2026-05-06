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

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
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
inline constexpr bool transform_epilog_should_assume_aligned_input =
  ::cuda::std::is_pointer_v<::cuda::std::decay_t<InputIteratorT>>
  && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<
    ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::decay_t<InputIteratorT>>>>;

template <typename InputIteratorT>
using transform_epilog_kernel_input_t = ::cuda::std::conditional_t<
  transform_epilog_should_assume_aligned_input<InputIteratorT>,
  transform_epilog_aligned_base_ptr<::cuda::std::remove_pointer_t<::cuda::std::decay_t<InputIteratorT>>>,
  ::cuda::std::decay_t<InputIteratorT>>;

template <typename InputIteratorT>
struct transform_epilog_input_tuple
{
  using type = ::cuda::std::tuple<transform_epilog_kernel_input_t<InputIteratorT>>;
};

template <typename... InputIteratorTs>
struct transform_epilog_input_tuple<::cuda::std::tuple<InputIteratorTs...>>
{
  using type = ::cuda::std::tuple<transform_epilog_kernel_input_t<InputIteratorTs>...>;
};

template <typename InputIteratorT>
using transform_epilog_input_tuple_t =
  typename transform_epilog_input_tuple<::cuda::std::decay_t<InputIteratorT>>::type;

template <typename InputIteratorT>
struct transform_epilog_tuple_all_load_to_shared : ::cuda::std::false_type
{};

template <typename... InputIteratorTs>
struct transform_epilog_tuple_all_load_to_shared<::cuda::std::tuple<InputIteratorTs...>>
    : ::cuda::std::bool_constant<transform_epilog_all_load_to_shared<InputIteratorTs...>>
{};

template <typename InputIteratorT, typename DeviceEpilogOpT>
inline constexpr bool transform_epilog_can_use_flat_no_epilog =
  ::cuda::std::is_same_v<::cuda::std::decay_t<DeviceEpilogOpT>, NoopDeviceEpilogOp>
  && transform_epilog_tuple_all_load_to_shared<::cuda::std::decay_t<InputIteratorT>>::value;

template <typename InputIteratorT, int ItemsPerThread, typename DeviceEpilogOpT>
inline constexpr bool transform_epilog_can_use_static_segment_32 =
  transform_epilog_tuple_all_load_to_shared<::cuda::std::decay_t<InputIteratorT>>::value
  && ((32 % ItemsPerThread == 0) || transform_epilog_can_use_flat_no_epilog<InputIteratorT, DeviceEpilogOpT>);

template <typename KernelSourceT, typename = void>
struct transform_epilog_has_segment_32_kernel : ::cuda::std::false_type
{};

template <typename KernelSourceT>
struct transform_epilog_has_segment_32_kernel<
  KernelSourceT,
  ::cuda::std::void_t<decltype(::cuda::std::declval<KernelSourceT>().HierarchicalTransformEpilogKernelSegment32())>>
    : ::cuda::std::true_type
{};

template <typename InputIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto transform_epilog_make_kernel_input(InputIteratorT&& d_in)
{
  if constexpr (transform_epilog_should_assume_aligned_input<InputIteratorT>)
  {
    return transform_epilog_make_aligned_base_ptr(d_in);
  }
  else
  {
    static_assert(transform_epilog_should_assume_aligned_input<InputIteratorT>,
                  "TransformEpilog requires all transform inputs to be raw pointers to trivially relocatable types.");
  }
}

template <typename InputIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto make_transform_epilog_input_tuple(InputIteratorT&& d_in)
{
  if constexpr (transform_epilog_tuple_all_load_to_shared<::cuda::std::decay_t<InputIteratorT>>::value)
  {
    return ::cuda::std::forward<InputIteratorT>(d_in);
  }
  else if constexpr (is_cuda_std_tuple<::cuda::std::decay_t<InputIteratorT>>::value)
  {
    return ::cuda::std::apply(
      []<typename... InputIteratorTs>(InputIteratorTs&&... input_iterators) {
        return ::cuda::std::make_tuple(
          transform_epilog_make_kernel_input(::cuda::std::forward<InputIteratorTs>(input_iterators))...);
      },
      ::cuda::std::forward<InputIteratorT>(d_in));
  }
  else
  {
    return ::cuda::std::make_tuple(transform_epilog_make_kernel_input(::cuda::std::forward<InputIteratorT>(d_in)));
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
  static_assert(transform_epilog_all_load_to_shared<InputIteratorTs...>,
                "TransformEpilog requires all transform inputs to be raw pointers to trivially relocatable types.");

  static constexpr bool use_striped_no_epilog =
    ::cuda::std::is_same_v<::cuda::std::decay_t<DeviceEpilogOpT>, NoopDeviceEpilogOp>
    && transform_epilog_all_load_to_shared<InputIteratorTs...>;

  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformEpilogKernel,
    DeviceHierarchicalTransformEpilogKernel<
      BlockThreads,
      ItemsPerThread,
      0,
      use_striped_no_epilog,
      OutputIteratorT,
      TransformOpT,
      DeviceEpilogOpT,
      InputIteratorTs...>)
  CUB_DEFINE_KERNEL_GETTER(
    HierarchicalTransformEpilogKernelSegment32,
    DeviceHierarchicalTransformEpilogKernel<
      BlockThreads,
      ItemsPerThread,
      32,
      use_striped_no_epilog,
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
  static_assert(ItemsPerThread > 0, "ItemsPerThread must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");
  static_assert(transform_epilog_tuple_all_load_to_shared<::cuda::std::decay_t<InputIteratorT>>::value,
                "TransformEpilog requires all transform inputs to be raw pointers to trivially relocatable types.");

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

  template <bool UseFlatTile, typename DeviceHierarchicalTransformEpilogKernelT, ::cuda::std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokeKernel(
    DeviceHierarchicalTransformEpilogKernelT hierarchical_transform_epilog_kernel, ::cuda::std::index_sequence<Is...>)
  {
    if (num_segments == 0)
    {
      return cudaSuccess;
    }

    constexpr int warp_threads = 32;

    if (num_segments < 0 || segment_size <= 0 || segment_size > warp_threads)
    {
      return cudaErrorInvalidValue;
    }

    constexpr auto max_total_items = static_cast<::cuda::std::int64_t>((::cuda::std::numeric_limits<int>::max)());

    if (num_segments > max_total_items / segment_size)
    {
      return cudaErrorInvalidValue;
    }

    const int total_items = static_cast<int>(num_segments * static_cast<::cuda::std::int64_t>(segment_size));

    int required_blocks = 0;
    if constexpr (UseFlatTile)
    {
      required_blocks = ::cuda::ceil_div(total_items, BlockThreads * ItemsPerThread);
    }
    else
    {
      if (segment_size % ItemsPerThread != 0)
      {
        return cudaErrorInvalidValue;
      }

      const int threads_per_segment = segment_size / ItemsPerThread;
      if (threads_per_segment <= 0 || threads_per_segment > BlockThreads)
      {
        return cudaErrorInvalidValue;
      }

      const int segments_per_block = BlockThreads / threads_per_segment;
      if (segments_per_block <= 0)
      {
        return cudaErrorInvalidValue;
      }

      required_blocks = ::cuda::ceil_div(static_cast<int>(num_segments), segments_per_block);
    }

    int max_grid_dim_x = 0;
    if (const auto error = CubDebug(launcher_factory.MaxGridDimX(max_grid_dim_x)))
    {
      return error;
    }

    const bool aligned_inputs_are_valid =
      [&]<::cuda::std::size_t... InputIndices>(::cuda::std::index_sequence<InputIndices...>) {
        return (... && transform_epilog_is_valid_aligned_input(::cuda::std::get<InputIndices>(d_in), total_items));
      }(::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<InputIteratorT>>{});

    if (!aligned_inputs_are_valid)
    {
      return cudaErrorInvalidValue;
    }

    if (required_blocks > max_grid_dim_x)
    {
      return cudaErrorInvalidValue;
    }

    launcher_factory(required_blocks, BlockThreads, 0, stream)
      .doit(hierarchical_transform_epilog_kernel,
            ::cuda::std::get<Is>(d_in)...,
            d_out,
            total_items,
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
    using input_indices_t = ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<InputIteratorT>>;

    if constexpr (transform_epilog_can_use_static_segment_32<InputIteratorT, ItemsPerThread, DeviceEpilogOpT>
                  && transform_epilog_has_segment_32_kernel<KernelSource>::value)
    {
      if (segment_size == 32)
      {
        constexpr bool use_flat_tile = transform_epilog_can_use_flat_no_epilog<InputIteratorT, DeviceEpilogOpT>;
        return InvokeKernel<use_flat_tile>(
          kernel_source.HierarchicalTransformEpilogKernelSegment32(), input_indices_t{});
      }
    }

    return InvokeKernel<false>(kernel_source.HierarchicalTransformEpilogKernel(), input_indices_t{});
  }
};

template <int BlockThreads   = 256,
          int ItemsPerThread = 5,
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
  static_assert(transform_epilog_tuple_all_load_to_shared<InputIteratorTupleT>::value,
                "TransformEpilog requires all transform inputs to be raw pointers to trivially relocatable types.");

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
  static_assert(transform_epilog_tuple_all_load_to_shared<InputIteratorTupleT>::value,
                "TransformEpilog requires all transform inputs to be raw pointers to trivially relocatable types.");

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
