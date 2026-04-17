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

#include <cub/block/block_load.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <typename T, typename = void>
struct is_tuple_like : ::cuda::std::false_type
{};

template <typename T>
struct is_tuple_like<T, ::cuda::std::void_t<decltype(::cuda::std::tuple_size<T>::value)>> : ::cuda::std::true_type
{};

template <typename InputIteratorT, ::cuda::std::size_t... Indices>
_CCCL_DEVICE _CCCL_FORCEINLINE auto load_transform_epilog_input_impl(
  InputIteratorT const& d_in, ::cuda::std::int64_t item_index, ::cuda::std::index_sequence<Indices...>)
{
  return ::cuda::std::make_tuple(::cuda::std::get<Indices>(d_in)[item_index]...);
}

template <typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
load_transform_epilog_input(InputIteratorT const& d_in, ::cuda::std::int64_t item_index)
{
  if constexpr (is_tuple_like<InputIteratorT>::value)
  {
    return load_transform_epilog_input_impl(
      d_in, item_index, ::cuda::std::make_index_sequence<::cuda::std::tuple_size<InputIteratorT>::value>{});
  }
  else
  {
    return *(d_in + item_index);
  }
}

template <typename T, int ItemsPerThread>
struct transform_epilog_thread_items
{
  T items[ItemsPerThread];
};

template <int BlockThreads, int ItemsPerThread, typename IteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
load_transform_epilog_blocked_items(IteratorT const& d_in, ::cuda::std::int64_t tile_item_base, int tile_items)
{
  using value_t = cub::detail::it_value_t<IteratorT>;
  transform_epilog_thread_items<value_t, ItemsPerThread> loaded{};
  cub::BlockLoad<value_t, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_STRIPED>{}.Load(
    d_in + tile_item_base, loaded.items, tile_items);
  return loaded;
}

template <int BlockThreads, int ItemsPerThread, typename InputIteratorT, ::cuda::std::size_t... Indices>
_CCCL_DEVICE _CCCL_FORCEINLINE auto load_transform_epilog_blocked_inputs_impl(
  InputIteratorT const& d_in,
  ::cuda::std::int64_t tile_item_base,
  int tile_items,
  ::cuda::std::index_sequence<Indices...>)
{
  return ::cuda::std::make_tuple(load_transform_epilog_blocked_items<BlockThreads, ItemsPerThread>(
    ::cuda::std::get<Indices>(d_in), tile_item_base, tile_items)...);
}

template <int BlockThreads, int ItemsPerThread, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
load_transform_epilog_blocked_inputs(InputIteratorT const& d_in, ::cuda::std::int64_t tile_item_base, int tile_items)
{
  return load_transform_epilog_blocked_inputs_impl<BlockThreads, ItemsPerThread>(
    d_in,
    tile_item_base,
    tile_items,
    ::cuda::std::make_index_sequence<::cuda::std::tuple_size<InputIteratorT>::value>{});
}

template <int ItemsPerThread, typename LoadedInputsT, ::cuda::std::size_t... Indices>
_CCCL_DEVICE _CCCL_FORCEINLINE auto make_transform_epilog_input_from_blocked_items_impl(
  LoadedInputsT const& loaded_inputs, int item_slot, ::cuda::std::index_sequence<Indices...>)
{
  return ::cuda::std::make_tuple(::cuda::std::get<Indices>(loaded_inputs).items[item_slot]...);
}

template <int ItemsPerThread, typename LoadedInputsT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
make_transform_epilog_input_from_blocked_items(LoadedInputsT const& loaded_inputs, int item_slot)
{
  return make_transform_epilog_input_from_blocked_items_impl<ItemsPerThread>(
    loaded_inputs, item_slot, ::cuda::std::make_index_sequence<::cuda::std::tuple_size<LoadedInputsT>::value>{});
}

template <typename TransformOpT, typename InputRefT, bool AcceptsIndex>
struct transform_epilog_result
{};

template <typename TransformOpT, typename InputRefT>
struct transform_epilog_result<TransformOpT, InputRefT, true>
{
  using type = ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, ::cuda::std::int64_t, InputRefT>>;
};

template <typename TransformOpT, typename InputRefT>
struct transform_epilog_result<TransformOpT, InputRefT, false>
{
  using type = ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, InputRefT>>;
};

template <int BlockThreads,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformEpilogKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const ::cuda::std::int64_t num_segments,
  _CCCL_GRID_CONSTANT const int segment_size,
  TransformOpT transform_op,
  DeviceEpilogOpT device_epilog_op)
{
  // Initial warp-only implementation:
  // - each warp owns one segment per loop iteration
  // - each thread computes one logical transform result for its lane's item
  // - segment size is provided explicitly and must fit in one warp
  // - `transform_op(index, value)` or `transform_op(value)` returns the per-item result
  // - the transform result is materialized immediately through the output iterator
  // - the same transform result is forwarded to the cooperative epilog
  // - `device_epilog_op` owns all per-segment and device-wide side effects, such as writing a mask word or updating a
  //   global counter

  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");

  using input_ref_t = decltype(load_transform_epilog_input(::cuda::std::declval<InputIteratorT const&>(), 0));
  constexpr bool transform_accepts_index = ::cuda::std::is_invocable_v<TransformOpT, ::cuda::std::int64_t, input_ref_t>;

  static_assert(transform_accepts_index || ::cuda::std::is_invocable_v<TransformOpT, input_ref_t>,
                "transform_op must be invocable with either (index, value) or (value).");

  using transform_result_t = typename transform_epilog_result<TransformOpT, input_ref_t, transform_accepts_index>::type;

  static_assert(::cuda::std::is_default_constructible_v<transform_result_t>,
                "The current hierarchical transform epilog kernel default-initializes the per-thread transform result "
                "for out-of-range lanes.");

  constexpr int warp_threads      = 32;
  constexpr int warps_per_block   = BlockThreads / warp_threads;
  constexpr int items_per_thread  = 4;
  constexpr int tile_items        = BlockThreads * items_per_thread;
  constexpr int segments_per_tile = warps_per_block * items_per_thread;

  const int lane_rank            = static_cast<int>(threadIdx.x % warp_threads);
  const int warp_rank_in_block   = static_cast<int>(threadIdx.x / warp_threads);
  const auto block_hierarchy     = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto warp_group                = ::cuda::experimental::this_warp{block_hierarchy};
  const auto global_segment_rank = static_cast<::cuda::std::int64_t>(blockIdx.x) * warps_per_block + warp_rank_in_block;
  const auto segment_stride      = static_cast<::cuda::std::int64_t>(gridDim.x) * warps_per_block;

  if constexpr (is_tuple_like<InputIteratorT>::value)
  {
    if (segment_size == warp_threads)
    {
      using transform_result_array_t = transform_epilog_thread_items<transform_result_t, items_per_thread>;

      const auto tile_segment_stride      = static_cast<::cuda::std::int64_t>(gridDim.x) * segments_per_tile;
      const auto global_tile_segment_rank = static_cast<::cuda::std::int64_t>(blockIdx.x) * segments_per_tile;

      for (::cuda::std::int64_t tile_segment_index = global_tile_segment_rank; tile_segment_index < num_segments;
           tile_segment_index += tile_segment_stride)
      {
        const auto tile_item_base  = tile_segment_index * static_cast<::cuda::std::int64_t>(segment_size);
        const auto total_items     = num_segments * static_cast<::cuda::std::int64_t>(segment_size);
        const auto remaining_items = total_items - tile_item_base;
        const int valid_tile_items = remaining_items < tile_items ? static_cast<int>(remaining_items) : tile_items;

        auto loaded_inputs =
          load_transform_epilog_blocked_inputs<BlockThreads, items_per_thread>(d_in, tile_item_base, valid_tile_items);
        transform_result_array_t thread_results{};

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item_slot = 0; item_slot < items_per_thread; ++item_slot)
        {
          const int local_item_index = static_cast<int>(threadIdx.x) + item_slot * BlockThreads;
          if (local_item_index < valid_tile_items)
          {
            const auto item_index = tile_item_base + local_item_index;
            auto input_value =
              make_transform_epilog_input_from_blocked_items<items_per_thread>(loaded_inputs, item_slot);
            if constexpr (transform_accepts_index)
            {
              thread_results.items[item_slot] = transform_op(item_index, input_value);
            }
            else
            {
              thread_results.items[item_slot] = transform_op(input_value);
            }
            *(d_out + item_index) = thread_results.items[item_slot];
          }
        }

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item_slot = 0; item_slot < items_per_thread; ++item_slot)
        {
          const auto segment_index =
            tile_segment_index + warp_rank_in_block + static_cast<::cuda::std::int64_t>(item_slot) * warps_per_block;
          if (segment_index < num_segments)
          {
            device_epilog_op(warp_group, segment_index, thread_results.items[item_slot]);
          }
        }
      }
      return;
    }
  }

  for (::cuda::std::int64_t segment_index = global_segment_rank; segment_index < num_segments;
       segment_index += segment_stride)
  {
    const auto item_index = segment_index * static_cast<::cuda::std::int64_t>(segment_size) + lane_rank;
    transform_result_t thread_result{};

    if (lane_rank < segment_size)
    {
      auto input_value = load_transform_epilog_input(d_in, item_index);
      if constexpr (transform_accepts_index)
      {
        thread_result = transform_op(item_index, input_value);
      }
      else
      {
        thread_result = transform_op(input_value);
      }
      *(d_out + item_index) = thread_result;
    }

    device_epilog_op(warp_group, segment_index, thread_result);
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
