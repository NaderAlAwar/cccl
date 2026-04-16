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

#include <cub/block/block_load_to_shared.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/span>
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

template <typename T>
struct is_raw_pointer_tuple : ::cuda::std::false_type
{};

template <typename LhsT, typename RhsT, typename PredT>
struct is_raw_pointer_tuple<::cuda::std::tuple<LhsT*, RhsT*, PredT*>>
    : ::cuda::std::bool_constant<::cuda::std::is_same_v<PredT, bool>>
{};

template <typename LhsT, typename RhsT, typename PredT>
struct is_raw_pointer_tuple<::cuda::std::tuple<const LhsT*, const RhsT*, const PredT*>>
    : ::cuda::std::bool_constant<::cuda::std::is_same_v<PredT, bool>>
{};

template <typename T>
struct is_nullable_raw_pointer_mask_tuple : ::cuda::std::false_type
{};

template <typename LhsT, typename LhsMaskT, typename RhsT, typename RhsMaskT, typename PredT, typename PredMaskT>
struct is_nullable_raw_pointer_mask_tuple<
  ::cuda::std::tuple<const LhsT*, const LhsMaskT*, const RhsT*, const RhsMaskT*, const PredT*, const PredMaskT*>>
    : ::cuda::std::bool_constant<::cuda::std::is_same_v<PredT, bool>&& ::cuda::std::is_integral_v<
        LhsMaskT>&& ::cuda::std::is_integral_v<RhsMaskT>&& ::cuda::std::is_integral_v<PredMaskT>>
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

template <typename TransformOpT, typename LhsValueT, typename RhsValueT, typename PredValueT, bool AcceptsIndex>
struct nullable_transform_epilog_result
{};

template <typename TransformOpT, typename LhsValueT, typename RhsValueT, typename PredValueT>
struct nullable_transform_epilog_result<TransformOpT, LhsValueT, RhsValueT, PredValueT, true>
{
  using type = ::cuda::std::decay_t<
    ::cuda::std::invoke_result_t<TransformOpT, ::cuda::std::int64_t, LhsValueT, RhsValueT, PredValueT, bool, bool, bool>>;
};

template <typename TransformOpT, typename LhsValueT, typename RhsValueT, typename PredValueT>
struct nullable_transform_epilog_result<TransformOpT, LhsValueT, RhsValueT, PredValueT, false>
{
  using type =
    ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, LhsValueT, RhsValueT, PredValueT, bool, bool, bool>>;
};

template <int BlockThreads,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformEpilogRawPointerKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const ::cuda::std::int64_t num_segments,
  _CCCL_GRID_CONSTANT const int segment_size,
  TransformOpT transform_op,
  DeviceEpilogOpT device_epilog_op)
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");
  static_assert(is_raw_pointer_tuple<InputIteratorT>::value,
                "The raw-pointer hierarchical transform epilog kernel requires a tuple of raw pointers.");

  using input_ref_t = decltype(load_transform_epilog_input(::cuda::std::declval<InputIteratorT const&>(), 0));
  constexpr bool transform_accepts_index = ::cuda::std::is_invocable_v<TransformOpT, ::cuda::std::int64_t, input_ref_t>;

  static_assert(transform_accepts_index || ::cuda::std::is_invocable_v<TransformOpT, input_ref_t>,
                "transform_op must be invocable with either (index, value) or (value).");

  using transform_result_t = typename transform_epilog_result<TransformOpT, input_ref_t, transform_accepts_index>::type;

  static_assert(::cuda::std::is_default_constructible_v<transform_result_t>,
                "The raw-pointer hierarchical transform epilog kernel default-initializes the per-thread transform "
                "result for out-of-range lanes.");

  constexpr int warp_threads      = 32;
  constexpr int warps_per_block   = BlockThreads / warp_threads;
  constexpr int items_per_thread  = 4;
  constexpr int tile_items        = BlockThreads * items_per_thread;
  constexpr int segments_per_tile = warps_per_block * items_per_thread;

  using lhs_ptr_t    = ::cuda::std::tuple_element_t<0, InputIteratorT>;
  using rhs_ptr_t    = ::cuda::std::tuple_element_t<1, InputIteratorT>;
  using pred_ptr_t   = ::cuda::std::tuple_element_t<2, InputIteratorT>;
  using lhs_value_t  = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<lhs_ptr_t>>;
  using rhs_value_t  = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<rhs_ptr_t>>;
  using pred_value_t = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<pred_ptr_t>>;

  using load2sh_t = cub::detail::BlockLoadToShared<BlockThreads>;

  struct alignas(cub::detail::LoadToSharedBufferAlignBytes<lhs_value_t>()) lhs_buffer_t
  {
    char storage[cub::detail::LoadToSharedBufferSizeBytes<lhs_value_t>(tile_items)];
  };

  struct alignas(cub::detail::LoadToSharedBufferAlignBytes<rhs_value_t>()) rhs_buffer_t
  {
    char storage[cub::detail::LoadToSharedBufferSizeBytes<rhs_value_t>(tile_items)];
  };

  struct alignas(cub::detail::LoadToSharedBufferAlignBytes<pred_value_t>()) pred_buffer_t
  {
    char storage[cub::detail::LoadToSharedBufferSizeBytes<pred_value_t>(tile_items)];
  };

  struct raw_pointer_temp_storage
  {
    typename load2sh_t::TempStorage load2sh;
    lhs_buffer_t lhs_buffer;
    rhs_buffer_t rhs_buffer;
    pred_buffer_t pred_buffer;
  };

  __shared__ raw_pointer_temp_storage temp_storage;

  const int warp_rank_in_block = static_cast<int>(threadIdx.x / warp_threads);
  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto warp_group              = ::cuda::experimental::this_warp{block_hierarchy};
  load2sh_t load2sh{temp_storage.load2sh};

  using transform_result_array_t = transform_epilog_thread_items<transform_result_t, items_per_thread>;

  const auto lhs_ptr  = ::cuda::std::get<0>(d_in);
  const auto rhs_ptr  = ::cuda::std::get<1>(d_in);
  const auto pred_ptr = ::cuda::std::get<2>(d_in);

  const auto tile_segment_stride      = static_cast<::cuda::std::int64_t>(gridDim.x) * segments_per_tile;
  const auto global_tile_segment_rank = static_cast<::cuda::std::int64_t>(blockIdx.x) * segments_per_tile;

  for (::cuda::std::int64_t tile_segment_index = global_tile_segment_rank; tile_segment_index < num_segments;
       tile_segment_index += tile_segment_stride)
  {
    const auto tile_item_base  = tile_segment_index * static_cast<::cuda::std::int64_t>(segment_size);
    const auto total_items     = num_segments * static_cast<::cuda::std::int64_t>(segment_size);
    const auto remaining_items = total_items - tile_item_base;
    const int valid_tile_items = remaining_items < tile_items ? static_cast<int>(remaining_items) : tile_items;
    transform_result_array_t thread_results{};

    auto lhs_items = load2sh.CopyAsync(
      ::cuda::std::span<char>{temp_storage.lhs_buffer.storage, sizeof(temp_storage.lhs_buffer.storage)},
      ::cuda::std::span<const lhs_value_t>{
        lhs_ptr + tile_item_base, static_cast<::cuda::std::size_t>(valid_tile_items)});
    auto rhs_items = load2sh.CopyAsync(
      ::cuda::std::span<char>{temp_storage.rhs_buffer.storage, sizeof(temp_storage.rhs_buffer.storage)},
      ::cuda::std::span<const rhs_value_t>{
        rhs_ptr + tile_item_base, static_cast<::cuda::std::size_t>(valid_tile_items)});
    auto pred_items = load2sh.CopyAsync(
      ::cuda::std::span<char>{temp_storage.pred_buffer.storage, sizeof(temp_storage.pred_buffer.storage)},
      ::cuda::std::span<const pred_value_t>{
        pred_ptr + tile_item_base, static_cast<::cuda::std::size_t>(valid_tile_items)});
    auto token = load2sh.Commit();
    load2sh.Wait(::cuda::std::move(token));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item_slot = 0; item_slot < items_per_thread; ++item_slot)
    {
      const int local_item_index = static_cast<int>(threadIdx.x) + item_slot * BlockThreads;
      if (local_item_index < valid_tile_items)
      {
        const auto item_index = tile_item_base + local_item_index;
        auto input_value      = ::cuda::std::make_tuple(
          lhs_items[local_item_index], rhs_items[local_item_index], pred_items[local_item_index]);
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
}

template <int BlockThreads,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformEpilogNullableRawPointerKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const ::cuda::std::int64_t num_segments,
  _CCCL_GRID_CONSTANT const int segment_size,
  TransformOpT transform_op,
  DeviceEpilogOpT device_epilog_op)
{
  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");
  static_assert(is_nullable_raw_pointer_mask_tuple<InputIteratorT>::value,
                "The nullable raw-pointer hierarchical transform epilog kernel requires a tuple of raw pointers and "
                "mask pointers.");

  using lhs_value_t =
    ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::tuple_element_t<0, InputIteratorT>>>;
  using rhs_value_t =
    ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::tuple_element_t<2, InputIteratorT>>>;
  using pred_value_t =
    ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::tuple_element_t<4, InputIteratorT>>>;
  constexpr bool transform_accepts_index = ::cuda::std::
    is_invocable_v<TransformOpT, ::cuda::std::int64_t, lhs_value_t, rhs_value_t, pred_value_t, bool, bool, bool>;

  static_assert(
    transform_accepts_index
      || ::cuda::std::is_invocable_v<TransformOpT, lhs_value_t, rhs_value_t, pred_value_t, bool, bool, bool>,
    "transform_op must be invocable with either (index, lhs, rhs, pred, lhs_valid, rhs_valid, pred_valid) or "
    "(lhs, rhs, pred, lhs_valid, rhs_valid, pred_valid).");

  using transform_result_t = typename nullable_transform_epilog_result<
    TransformOpT,
    lhs_value_t,
    rhs_value_t,
    pred_value_t,
    transform_accepts_index>::type;

  static_assert(::cuda::std::is_default_constructible_v<transform_result_t>,
                "The nullable raw-pointer hierarchical transform epilog kernel default-initializes the per-thread "
                "transform result for out-of-range lanes.");

  constexpr int warp_threads      = 32;
  constexpr int warps_per_block   = BlockThreads / warp_threads;
  constexpr int items_per_thread  = 4;
  constexpr int tile_items        = BlockThreads * items_per_thread;
  constexpr int segments_per_tile = warps_per_block * items_per_thread;

  using lhs_ptr_t       = ::cuda::std::tuple_element_t<0, InputIteratorT>;
  using lhs_mask_ptr_t  = ::cuda::std::tuple_element_t<1, InputIteratorT>;
  using rhs_ptr_t       = ::cuda::std::tuple_element_t<2, InputIteratorT>;
  using rhs_mask_ptr_t  = ::cuda::std::tuple_element_t<3, InputIteratorT>;
  using pred_ptr_t      = ::cuda::std::tuple_element_t<4, InputIteratorT>;
  using pred_mask_ptr_t = ::cuda::std::tuple_element_t<5, InputIteratorT>;
  using mask_word_t     = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<lhs_mask_ptr_t>>;

  static_assert(sizeof(mask_word_t) <= sizeof(::cuda::std::uint32_t),
                "Nullable raw-pointer hierarchical transform epilog kernel currently expects 32-bit-compatible mask "
                "words.");

  using load2sh_t = cub::detail::BlockLoadToShared<BlockThreads>;

  struct alignas(cub::detail::LoadToSharedBufferAlignBytes<lhs_value_t>()) lhs_buffer_t
  {
    char storage[cub::detail::LoadToSharedBufferSizeBytes<lhs_value_t>(tile_items)];
  };

  struct alignas(cub::detail::LoadToSharedBufferAlignBytes<rhs_value_t>()) rhs_buffer_t
  {
    char storage[cub::detail::LoadToSharedBufferSizeBytes<rhs_value_t>(tile_items)];
  };

  struct alignas(cub::detail::LoadToSharedBufferAlignBytes<pred_value_t>()) pred_buffer_t
  {
    char storage[cub::detail::LoadToSharedBufferSizeBytes<pred_value_t>(tile_items)];
  };

  struct nullable_raw_pointer_temp_storage
  {
    typename load2sh_t::TempStorage load2sh;
    lhs_buffer_t lhs_buffer;
    rhs_buffer_t rhs_buffer;
    pred_buffer_t pred_buffer;
  };

  __shared__ nullable_raw_pointer_temp_storage temp_storage;

  const int lane_rank          = static_cast<int>(threadIdx.x % warp_threads);
  const int warp_rank_in_block = static_cast<int>(threadIdx.x / warp_threads);
  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto warp_group              = ::cuda::experimental::this_warp{block_hierarchy};
  load2sh_t load2sh{temp_storage.load2sh};

  using transform_result_array_t = transform_epilog_thread_items<transform_result_t, items_per_thread>;

  const auto lhs_ptr       = ::cuda::std::get<0>(d_in);
  const auto lhs_mask_ptr  = ::cuda::std::get<1>(d_in);
  const auto rhs_ptr       = ::cuda::std::get<2>(d_in);
  const auto rhs_mask_ptr  = ::cuda::std::get<3>(d_in);
  const auto pred_ptr      = ::cuda::std::get<4>(d_in);
  const auto pred_mask_ptr = ::cuda::std::get<5>(d_in);

  const auto tile_segment_stride      = static_cast<::cuda::std::int64_t>(gridDim.x) * segments_per_tile;
  const auto global_tile_segment_rank = static_cast<::cuda::std::int64_t>(blockIdx.x) * segments_per_tile;

  for (::cuda::std::int64_t tile_segment_index = global_tile_segment_rank; tile_segment_index < num_segments;
       tile_segment_index += tile_segment_stride)
  {
    const auto tile_item_base  = tile_segment_index * static_cast<::cuda::std::int64_t>(segment_size);
    const auto total_items     = num_segments * static_cast<::cuda::std::int64_t>(segment_size);
    const auto remaining_items = total_items - tile_item_base;
    const int valid_tile_items = remaining_items < tile_items ? static_cast<int>(remaining_items) : tile_items;
    transform_result_array_t thread_results{};

    auto lhs_items = load2sh.CopyAsync(
      ::cuda::std::span<char>{temp_storage.lhs_buffer.storage, sizeof(temp_storage.lhs_buffer.storage)},
      ::cuda::std::span<const lhs_value_t>{
        lhs_ptr + tile_item_base, static_cast<::cuda::std::size_t>(valid_tile_items)});
    auto rhs_items = load2sh.CopyAsync(
      ::cuda::std::span<char>{temp_storage.rhs_buffer.storage, sizeof(temp_storage.rhs_buffer.storage)},
      ::cuda::std::span<const rhs_value_t>{
        rhs_ptr + tile_item_base, static_cast<::cuda::std::size_t>(valid_tile_items)});
    auto pred_items = load2sh.CopyAsync(
      ::cuda::std::span<char>{temp_storage.pred_buffer.storage, sizeof(temp_storage.pred_buffer.storage)},
      ::cuda::std::span<const pred_value_t>{
        pred_ptr + tile_item_base, static_cast<::cuda::std::size_t>(valid_tile_items)});
    auto token = load2sh.Commit();
    load2sh.Wait(::cuda::std::move(token));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item_slot = 0; item_slot < items_per_thread; ++item_slot)
    {
      const int local_item_index = static_cast<int>(threadIdx.x) + item_slot * BlockThreads;
      const auto segment_index =
        tile_segment_index + warp_rank_in_block + static_cast<::cuda::std::int64_t>(item_slot) * warps_per_block;
      if (segment_index < num_segments)
      {
        const ::cuda::std::uint32_t lhs_word_root =
          lane_rank == 0
            ? static_cast<::cuda::std::uint32_t>(lhs_mask_ptr == nullptr ? ~0u : lhs_mask_ptr[segment_index])
            : 0u;
        const ::cuda::std::uint32_t rhs_word_root =
          lane_rank == 0
            ? static_cast<::cuda::std::uint32_t>(rhs_mask_ptr == nullptr ? ~0u : rhs_mask_ptr[segment_index])
            : 0u;
        const ::cuda::std::uint32_t pred_word_root =
          lane_rank == 0
            ? static_cast<::cuda::std::uint32_t>(pred_mask_ptr == nullptr ? ~0u : pred_mask_ptr[segment_index])
            : 0u;
        const ::cuda::std::uint32_t lhs_word  = __shfl_sync(0xFFFF'FFFFu, lhs_word_root, 0);
        const ::cuda::std::uint32_t rhs_word  = __shfl_sync(0xFFFF'FFFFu, rhs_word_root, 0);
        const ::cuda::std::uint32_t pred_word = __shfl_sync(0xFFFF'FFFFu, pred_word_root, 0);

        if (local_item_index < valid_tile_items)
        {
          const auto item_index = tile_item_base + local_item_index;
          const bool lhs_valid  = ((lhs_word >> lane_rank) & 1u) != 0;
          const bool rhs_valid  = ((rhs_word >> lane_rank) & 1u) != 0;
          const bool pred_valid = ((pred_word >> lane_rank) & 1u) != 0;
          const auto lhs_value  = lhs_items[local_item_index];
          const auto rhs_value  = rhs_items[local_item_index];
          const auto pred_value = pred_items[local_item_index];

          if constexpr (transform_accepts_index)
          {
            thread_results.items[item_slot] =
              transform_op(item_index, lhs_value, rhs_value, pred_value, lhs_valid, rhs_valid, pred_valid);
          }
          else
          {
            thread_results.items[item_slot] =
              transform_op(lhs_value, rhs_value, pred_value, lhs_valid, rhs_valid, pred_valid);
          }
          *(d_out + item_index) = thread_results.items[item_slot];
        }
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
}

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

  constexpr int warp_threads    = 32;
  constexpr int warps_per_block = BlockThreads / warp_threads;

  const int lane_rank            = static_cast<int>(threadIdx.x % warp_threads);
  const int warp_rank_in_block   = static_cast<int>(threadIdx.x / warp_threads);
  const auto block_hierarchy     = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto warp_group                = ::cuda::experimental::this_warp{block_hierarchy};
  const auto global_segment_rank = static_cast<::cuda::std::int64_t>(blockIdx.x) * warps_per_block + warp_rank_in_block;
  const auto segment_stride      = static_cast<::cuda::std::int64_t>(gridDim.x) * warps_per_block;

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
