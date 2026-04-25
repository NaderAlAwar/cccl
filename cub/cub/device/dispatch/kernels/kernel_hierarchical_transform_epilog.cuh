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

#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
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

template <int ItemsPerThread>
struct transform_epilog_indices_storage
{
  ::cuda::std::int64_t values[ItemsPerThread];

  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize()
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      values[item] = -1;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(int item, ::cuda::std::int64_t value)
  {
    values[item] = value;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE auto get() const -> const ::cuda::std::int64_t (&)[ItemsPerThread]
  {
    return values;
  }
};

struct transform_epilog_no_indices_storage
{
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize() {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(int, ::cuda::std::int64_t) {}
};

template <int BlockThreads,
          int ItemsPerThread,
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
  // Current implementation:
  // - the core transform path is flat and row-oriented
  // - each block iteration covers a linear tile of BlockThreads * ItemsPerThread rows
  // - each thread owns ItemsPerThread consecutive rows inside that tile
  // - segment structure is preserved only through the per-item indices passed to the epilog
  // - `transform_op(index, value)` or `transform_op(value)` returns the per-item result
  // - the transform result is materialized immediately through the output iterator
  // - the same transform result is forwarded to the cooperative epilog
  // - `device_epilog_op` owns all per-segment and device-wide side effects, such as writing a mask word or updating a
  //   global counter

  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");

  using input_ref_t                      = cub::detail::it_reference_t<InputIteratorT>;
  constexpr bool transform_accepts_index = ::cuda::std::is_invocable_v<TransformOpT, ::cuda::std::int64_t, input_ref_t>;

  static_assert(transform_accepts_index || ::cuda::std::is_invocable_v<TransformOpT, input_ref_t>,
                "transform_op must be invocable with either (index, value) or (value).");

  using transform_result_t = typename transform_epilog_result<TransformOpT, input_ref_t, transform_accepts_index>::type;

  static_assert(::cuda::std::is_default_constructible_v<transform_result_t>,
                "The current hierarchical transform epilog kernel default-initializes the per-thread transform result "
                "for out-of-range lanes.");
  static_assert(::cuda::std::indirectly_writable<OutputIteratorT, const transform_result_t&>,
                "OutputIteratorT must be indirectly writable from the transform result type.");

  constexpr int warp_threads = 32;
  const auto block_hierarchy = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group           = ::cuda::experimental::this_block{block_hierarchy};
  using block_group_t        = decltype(block_group);
  using results_arg_t        = const transform_result_t(&)[ItemsPerThread];
  using indices_arg_t        = const ::cuda::std::int64_t (&)[ItemsPerThread];
  constexpr bool epilog_accepts_indices =
    ::cuda::std::is_invocable_v<DeviceEpilogOpT, block_group_t, results_arg_t, indices_arg_t>;

  static_assert(epilog_accepts_indices || ::cuda::std::is_invocable_v<DeviceEpilogOpT, block_group_t, results_arg_t>,
                "device_epilog_op must be invocable with either "
                "(block_group, results) or (block_group, results, indices).");

  using indices_storage_t =
    ::cuda::std::conditional_t<epilog_accepts_indices,
                               transform_epilog_indices_storage<ItemsPerThread>,
                               transform_epilog_no_indices_storage>;
  using block_load_t = BlockLoad<input_ref_t, BlockThreads, ItemsPerThread, BLOCK_LOAD_WARP_TRANSPOSE>;

  const auto total_items                  = num_segments * static_cast<::cuda::std::int64_t>(segment_size);
  constexpr int items_per_block_iteration = BlockThreads * ItemsPerThread;
  const auto block_item_stride            = static_cast<::cuda::std::int64_t>(gridDim.x) * items_per_block_iteration;
  const auto block_item_base              = static_cast<::cuda::std::int64_t>(blockIdx.x) * items_per_block_iteration;
  const auto linear_tid                   = static_cast<int>(threadIdx.x);
  __shared__ typename block_load_t::TempStorage temp_storage;

  for (::cuda::std::int64_t tile_item_base = block_item_base; tile_item_base < total_items;
       tile_item_base += block_item_stride)
  {
    input_ref_t loaded_values[ItemsPerThread];
    const bool full_tile = tile_item_base + items_per_block_iteration <= total_items;

    if (full_tile)
    {
      block_load_t(temp_storage).Load(d_in + tile_item_base, loaded_values);
    }
    else
    {
      const auto remaining_items = total_items - tile_item_base;
      block_load_t(temp_storage).Load(d_in + tile_item_base, loaded_values, static_cast<int>(remaining_items));
    }
    __syncthreads();

    transform_result_t output_values[ItemsPerThread]{};
    indices_storage_t indices_storage;
    const auto thread_item_base = tile_item_base + static_cast<::cuda::std::int64_t>(linear_tid) * ItemsPerThread;

    indices_storage.initialize();

    if (thread_item_base + ItemsPerThread <= total_items)
    {
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const auto item_index = thread_item_base + item;

        if constexpr (transform_accepts_index)
        {
          output_values[item] = transform_op(item_index, loaded_values[item]);
        }
        else
        {
          output_values[item] = transform_op(loaded_values[item]);
        }

        d_out[item_index] = output_values[item];
        indices_storage.set(item, item_index);
      }
    }
    else
    {
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const auto item_index = thread_item_base + item;
        if (item_index >= total_items)
        {
          continue;
        }

        if constexpr (transform_accepts_index)
        {
          output_values[item] = transform_op(item_index, loaded_values[item]);
        }
        else
        {
          output_values[item] = transform_op(loaded_values[item]);
        }

        d_out[item_index] = output_values[item];
        indices_storage.set(item, item_index);
      }
    }

    if constexpr (epilog_accepts_indices)
    {
      device_epilog_op(block_group, output_values, indices_storage.get());
    }
    else
    {
      device_epilog_op(block_group, output_values);
    }
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
