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
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <typename TransformOpT, bool AcceptsIndex, typename... InputValueTs>
struct transform_epilog_result
{};

template <typename TransformOpT, typename... InputValueTs>
struct transform_epilog_result<TransformOpT, true, InputValueTs...>
{
  using type = ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, ::cuda::std::int64_t, InputValueTs...>>;
};

template <typename TransformOpT, typename... InputValueTs>
struct transform_epilog_result<TransformOpT, false, InputValueTs...>
{
  using type = ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, InputValueTs...>>;
};

template <int ItemsPerThread, typename InputValueT>
struct transform_epilog_input_values
{
  InputValueT values[ItemsPerThread];
};

template <typename InputIteratorT>
inline constexpr BlockLoadAlgorithm transform_epilog_load_algorithm =
  ::cuda::std::is_pointer_v<::cuda::std::decay_t<InputIteratorT>> ? BLOCK_LOAD_DIRECT : BLOCK_LOAD_WARP_TRANSPOSE;

template <int BlockThreads, int ItemsPerThread, typename... InputIteratorTs>
using transform_epilog_block_load_storage_t =
  ::cuda::std::aligned_union_t<0,
                               typename BlockLoad<cub::detail::it_value_t<InputIteratorTs>,
                                                  BlockThreads,
                                                  ItemsPerThread,
                                                  transform_epilog_load_algorithm<InputIteratorTs>>::TempStorage...>;

template <typename BlockLoadStorageT, typename BlockLoadTempStorageT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto transform_epilog_alias_load_storage(BlockLoadStorageT& load_storage)
  -> BlockLoadTempStorageT&
{
  return reinterpret_cast<BlockLoadTempStorageT&>(load_storage);
}

template <int Index,
          int BlockThreads,
          int ItemsPerThread,
          typename BlockLoadStorageT,
          typename LoadedInputsT,
          typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_epilog_load_input(
  BlockLoadStorageT& load_storage,
  LoadedInputsT& loaded_inputs,
  InputIteratorT d_in,
  ::cuda::std::int64_t tile_item_base,
  bool full_tile,
  int remaining_items)
{
  using input_value_t = cub::detail::it_value_t<InputIteratorT>;
  using block_load_t =
    BlockLoad<input_value_t, BlockThreads, ItemsPerThread, transform_epilog_load_algorithm<InputIteratorT>>;

  auto& values = ::cuda::std::get<Index>(loaded_inputs).values;
  auto input   = d_in + tile_item_base;
  auto& storage =
    transform_epilog_alias_load_storage<BlockLoadStorageT, typename block_load_t::TempStorage>(load_storage);

  if (full_tile)
  {
    block_load_t(storage).Load(input, values);
  }
  else
  {
    block_load_t(storage).Load(input, values, remaining_items);
  }

  if constexpr (transform_epilog_load_algorithm<InputIteratorT> == BLOCK_LOAD_WARP_TRANSPOSE)
  {
    __syncthreads();
  }
}

template <int BlockThreads,
          int ItemsPerThread,
          typename BlockLoadStorageT,
          typename LoadedInputsT,
          ::cuda::std::size_t... Is,
          typename... InputIteratorTs>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_epilog_load_inputs(
  BlockLoadStorageT& load_storage,
  LoadedInputsT& loaded_inputs,
  ::cuda::std::index_sequence<Is...>,
  ::cuda::std::int64_t tile_item_base,
  bool full_tile,
  int remaining_items,
  InputIteratorTs... d_in)
{
  (transform_epilog_load_input<Is, BlockThreads, ItemsPerThread>(
     load_storage, loaded_inputs, d_in, tile_item_base, full_tile, remaining_items),
   ...);
}

template <bool TransformAcceptsIndex, typename TransformOpT, typename LoadedInputsT, ::cuda::std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE auto transform_epilog_invoke(
  TransformOpT& transform_op,
  ::cuda::std::int64_t item_index,
  LoadedInputsT& loaded_inputs,
  int item,
  ::cuda::std::index_sequence<Is...>)
{
  if constexpr (TransformAcceptsIndex)
  {
    return transform_op(item_index, ::cuda::std::get<Is>(loaded_inputs).values[item]...);
  }
  else
  {
    return transform_op(::cuda::std::get<Is>(loaded_inputs).values[item]...);
  }
}

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
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT,
          typename... InputIteratorTs>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformEpilogKernel(
  _CCCL_GRID_CONSTANT const InputIteratorTs... d_in,
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
  // - `transform_op(index, values...)` or `transform_op(values...)` returns the per-item result
  // - the transform result is materialized immediately through the output iterator
  // - the same transform result is forwarded to the cooperative epilog
  // - `device_epilog_op` owns all per-segment and device-wide side effects, such as writing a mask word or updating a
  //   global counter

  static_assert(BlockThreads > 0, "BlockThreads must be positive.");
  static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size.");
  static_assert(sizeof...(InputIteratorTs) > 0, "TransformEpilog requires at least one input.");

  constexpr bool transform_accepts_index =
    ::cuda::std::is_invocable_v<TransformOpT, ::cuda::std::int64_t, cub::detail::it_value_t<InputIteratorTs>...>;

  static_assert(
    transform_accepts_index || ::cuda::std::is_invocable_v<TransformOpT, cub::detail::it_value_t<InputIteratorTs>...>,
    "transform_op must be invocable with either (index, values...) or (values...).");

  using transform_result_t =
    typename transform_epilog_result<TransformOpT,
                                     transform_accepts_index,
                                     cub::detail::it_value_t<InputIteratorTs>...>::type;

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
  using input_load_storage_t = transform_epilog_block_load_storage_t<BlockThreads, ItemsPerThread, InputIteratorTs...>;
  using loaded_inputs_t =
    ::cuda::std::tuple<transform_epilog_input_values<ItemsPerThread, cub::detail::it_value_t<InputIteratorTs>>...>;
  using block_store_t = BlockStore<transform_result_t, BlockThreads, ItemsPerThread, BLOCK_STORE_WARP_TRANSPOSE>;

  const auto total_items                  = num_segments * static_cast<::cuda::std::int64_t>(segment_size);
  constexpr int items_per_block_iteration = BlockThreads * ItemsPerThread;
  const auto block_item_stride            = static_cast<::cuda::std::int64_t>(gridDim.x) * items_per_block_iteration;
  const auto block_item_base              = static_cast<::cuda::std::int64_t>(blockIdx.x) * items_per_block_iteration;
  const auto linear_tid                   = static_cast<int>(threadIdx.x);
  __shared__ union
  {
    input_load_storage_t load;
    typename block_store_t::TempStorage store;
  } temp_storage;

  for (::cuda::std::int64_t tile_item_base = block_item_base; tile_item_base < total_items;
       tile_item_base += block_item_stride)
  {
    loaded_inputs_t loaded_inputs;
    const bool full_tile       = tile_item_base + items_per_block_iteration <= total_items;
    const auto remaining_items = total_items - tile_item_base;

    transform_epilog_load_inputs<BlockThreads, ItemsPerThread>(
      temp_storage.load,
      loaded_inputs,
      ::cuda::std::index_sequence_for<InputIteratorTs...>{},
      tile_item_base,
      full_tile,
      static_cast<int>(remaining_items),
      d_in...);

    transform_result_t output_values[ItemsPerThread]{};
    indices_storage_t indices_storage;
    const auto thread_item_base = tile_item_base + static_cast<::cuda::std::int64_t>(linear_tid) * ItemsPerThread;

    indices_storage.initialize();

    if (thread_item_base + ItemsPerThread <= total_items)
    {
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const auto item_index = thread_item_base + item;

        output_values[item] = transform_epilog_invoke<transform_accepts_index>(
          transform_op, item_index, loaded_inputs, item, ::cuda::std::index_sequence_for<InputIteratorTs...>{});

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

        output_values[item] = transform_epilog_invoke<transform_accepts_index>(
          transform_op, item_index, loaded_inputs, item, ::cuda::std::index_sequence_for<InputIteratorTs...>{});

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

    if (full_tile)
    {
      block_store_t(temp_storage.store).Store(d_out + tile_item_base, output_values);
    }
    else
    {
      const auto remaining_items = total_items - tile_item_base;
      block_store_t(temp_storage.store).Store(d_out + tile_item_base, output_values, static_cast<int>(remaining_items));
    }
    __syncthreads();
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
