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

#include <cuda/__cmath/pow2.h>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
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

#if _CCCL_CTK_BELOW(13, 0)
struct alignas(32) transform_epilog_aligned32_t
{
  longlong4 data;
};
#endif // _CCCL_CTK_BELOW(13, 0)

template <int Bytes>
_CCCL_HOST_DEVICE _CCCL_CONSTEVAL auto transform_epilog_load_store_type()
{
  static_assert(::cuda::is_power_of_two(Bytes));
  if constexpr (Bytes == 1)
  {
    return ::cuda::std::int8_t{};
  }
  else if constexpr (Bytes == 2)
  {
    return ::cuda::std::int16_t{};
  }
  else if constexpr (Bytes == 4)
  {
    return ::cuda::std::int32_t{};
  }
  else if constexpr (Bytes == 8)
  {
    return ::cuda::std::int64_t{};
  }
  else if constexpr (Bytes == 16)
  {
    static_assert(alignof(int4) == 16);
    return int4{};
  }
  else if constexpr (Bytes == 32)
  {
#if _CCCL_CTK_BELOW(13, 0)
    static_assert(alignof(transform_epilog_aligned32_t) == 32);
    return transform_epilog_aligned32_t{};
#else // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^ / vvv _CCCL_CTK_AT_LEAST(13, 0) vvv
    return longlong4_32a{};
#endif // _CCCL_CTK_AT_LEAST(13, 0)
  }
  else
  {
    static_assert(Bytes <= 32, "Unsupported transform epilog vector load/store width.");
    return int4{};
  }
}

template <int ItemsPerThread>
_CCCL_HOST_DEVICE _CCCL_CONSTEVAL int transform_epilog_pack_size()
{
  return (ItemsPerThread % 4 == 0) ? 4 : (ItemsPerThread % 2 == 0) ? 2 : 1;
}

template <typename ValueT, int PackSize>
inline constexpr bool transform_epilog_supported_packed_type =
  ::cuda::is_power_of_two(sizeof(ValueT) * PackSize) && (sizeof(ValueT) * PackSize <= 32);

template <typename ValueT, int PackSize>
inline constexpr bool transform_epilog_pack_input_type =
  sizeof(::cuda::std::remove_cv_t<ValueT>) < sizeof(::cuda::std::int32_t)
  && !::cuda::std::is_same_v<::cuda::std::remove_cv_t<ValueT>, bool>
  && transform_epilog_supported_packed_type<::cuda::std::remove_cv_t<ValueT>, PackSize>;

template <typename ValueT, int PackSize>
inline constexpr bool transform_epilog_pack_output_type =
  sizeof(::cuda::std::remove_cv_t<ValueT>) < sizeof(::cuda::std::int32_t)
  && !::cuda::std::is_same_v<::cuda::std::remove_cv_t<ValueT>, bool>
  && transform_epilog_supported_packed_type<::cuda::std::remove_cv_t<ValueT>, PackSize>;

template <typename ValueT, int ItemsPerThread, int PackSize>
struct transform_epilog_packed_block_load_values
{
  static constexpr int packed_items_per_thread = ItemsPerThread / PackSize;
  using packed_t = decltype(transform_epilog_load_store_type<sizeof(ValueT) * PackSize>());

  packed_t values[packed_items_per_thread];

  _CCCL_DEVICE _CCCL_FORCEINLINE ValueT get(int item) const
  {
    return reinterpret_cast<const ValueT*>(values)[item];
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(int item, ValueT value)
  {
    reinterpret_cast<ValueT*>(values)[item] = value;
  }

  template <int BlockThreads>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load(const ValueT* input)
  {
    using block_load_t = BlockLoad<packed_t, BlockThreads, packed_items_per_thread, BLOCK_LOAD_VECTORIZE>;
    block_load_t().Load(reinterpret_cast<const packed_t*>(input), values);
  }

  template <int BlockThreads>
  _CCCL_DEVICE _CCCL_FORCEINLINE void store(ValueT* output)
  {
    using block_store_t = BlockStore<packed_t, BlockThreads, packed_items_per_thread, BLOCK_STORE_VECTORIZE>;
    block_store_t().Store(reinterpret_cast<packed_t*>(output), values);
  }
};

template <typename ValueT,
          int ItemsPerThread,
          int PackSize,
          bool UsePacked = transform_epilog_pack_input_type<ValueT, PackSize>>
struct transform_epilog_packed_block_input_values;

template <typename ValueT, int ItemsPerThread, int PackSize>
struct transform_epilog_packed_block_input_values<ValueT, ItemsPerThread, PackSize, true>
    : transform_epilog_packed_block_load_values<::cuda::std::remove_cv_t<ValueT>, ItemsPerThread, PackSize>
{};

template <typename ValueT, int ItemsPerThread, int PackSize>
struct transform_epilog_packed_block_input_values<ValueT, ItemsPerThread, PackSize, false>
{
  using value_t = ::cuda::std::remove_cv_t<ValueT>;

  value_t values[ItemsPerThread];

  _CCCL_DEVICE _CCCL_FORCEINLINE value_t get(int item) const
  {
    return values[item];
  }

  template <int BlockThreads>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load(const value_t* input)
  {
    using block_load_t = BlockLoad<value_t, BlockThreads, ItemsPerThread, BLOCK_LOAD_DIRECT>;
    block_load_t().Load(input, values);
  }
};

template <int PackSize, typename PointerT>
_CCCL_DEVICE _CCCL_FORCEINLINE bool transform_epilog_packed_aligned(PointerT ptr)
{
  using value_t  = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::decay_t<PointerT>>>;
  using packed_t = decltype(transform_epilog_load_store_type<sizeof(value_t) * PackSize>());
  return reinterpret_cast<::cuda::std::uintptr_t>(ptr) % alignof(packed_t) == 0;
}

template <int PackSize, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE bool
transform_epilog_input_packed_aligned(::cuda::std::int64_t tile_item_base, InputIteratorT d_in)
{
  using value_t = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::decay_t<InputIteratorT>>>;
  if constexpr (transform_epilog_pack_input_type<value_t, PackSize>)
  {
    return transform_epilog_packed_aligned<PackSize>(d_in + tile_item_base);
  }
  else
  {
    return true;
  }
}

template <int PackSize, typename OutputIteratorT, typename... InputIteratorTs>
_CCCL_DEVICE _CCCL_FORCEINLINE bool transform_epilog_packed_path_aligned(
  ::cuda::std::int64_t tile_item_base, OutputIteratorT d_out, InputIteratorTs... d_in)
{
  return transform_epilog_packed_aligned<PackSize>(d_out + tile_item_base)
      && (... && transform_epilog_input_packed_aligned<PackSize>(tile_item_base, d_in));
}

template <int BlockThreads,
          int ItemsPerThread,
          int PackSize,
          typename LoadedInputsT,
          ::cuda::std::size_t... Is,
          typename... InputIteratorTs>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_epilog_load_packed_block_inputs(
  LoadedInputsT& loaded_inputs,
  ::cuda::std::index_sequence<Is...>,
  ::cuda::std::int64_t tile_item_base,
  InputIteratorTs... d_in)
{
  (::cuda::std::get<Is>(loaded_inputs).template load<BlockThreads>(d_in + tile_item_base), ...);
}

template <int BlockThreads,
          int ItemsPerThread,
          int PackSize,
          typename OutputIteratorT,
          typename TransformOpT,
          ::cuda::std::size_t... Is,
          typename... InputIteratorTs>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_epilog_process_packed_block_tile(
  OutputIteratorT d_out,
  TransformOpT& transform_op,
  ::cuda::std::index_sequence<Is...> input_indices,
  ::cuda::std::int64_t tile_item_base,
  InputIteratorTs... d_in)
{
  using output_value_t = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::decay_t<OutputIteratorT>>>;
  using output_values_t = transform_epilog_packed_block_load_values<output_value_t, ItemsPerThread, PackSize>;
  using loaded_inputs_t = ::cuda::std::tuple<
    transform_epilog_packed_block_input_values<cub::detail::it_value_t<InputIteratorTs>, ItemsPerThread, PackSize>...>;

  loaded_inputs_t loaded_inputs;
  output_values_t output_values;

  transform_epilog_load_packed_block_inputs<BlockThreads, ItemsPerThread, PackSize>(
    loaded_inputs, input_indices, tile_item_base, d_in...);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    output_values.set(item, transform_op(::cuda::std::get<Is>(loaded_inputs).get(item)...));
  }

  output_values.template store<BlockThreads>(d_out + tile_item_base);
}

template <int ItemsPerThread,
          bool TransformAcceptsIndex,
          typename OutputIteratorT,
          typename TransformResultT,
          typename DeviceEpilogOpT,
          typename... InputIteratorTs>
inline constexpr bool transform_epilog_can_process_packed_block_path =
  transform_epilog_pack_size<ItemsPerThread>() > 1 && !TransformAcceptsIndex
  && ::cuda::std::is_pointer_v<::cuda::std::decay_t<OutputIteratorT>>
  && (... && ::cuda::std::is_pointer_v<::cuda::std::decay_t<InputIteratorTs>>)
  && ::cuda::std::is_same_v<::cuda::std::decay_t<DeviceEpilogOpT>, NoopDeviceEpilogOp>
  && ::cuda::std::is_same_v<
    ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<::cuda::std::decay_t<OutputIteratorT>>>,
    ::cuda::std::remove_cv_t<TransformResultT>>
  && ::cuda::std::is_trivially_copyable_v<::cuda::std::remove_cv_t<TransformResultT>>
  && (... && ::cuda::std::is_trivially_copyable_v<::cuda::std::remove_cv_t<cub::detail::it_value_t<InputIteratorTs>>>)
  && transform_epilog_pack_output_type<::cuda::std::remove_cv_t<TransformResultT>,
                                       transform_epilog_pack_size<ItemsPerThread>()>
  && (...
      || transform_epilog_pack_input_type<::cuda::std::remove_cv_t<cub::detail::it_value_t<InputIteratorTs>>,
                                          transform_epilog_pack_size<ItemsPerThread>()>);

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
  using block_store_t     = BlockStore<transform_result_t, BlockThreads, ItemsPerThread, BLOCK_STORE_WARP_TRANSPOSE>;
  constexpr int pack_size = transform_epilog_pack_size<ItemsPerThread>();
  constexpr bool can_process_packed_block_path = transform_epilog_can_process_packed_block_path<
    ItemsPerThread,
    transform_accepts_index,
    OutputIteratorT,
    transform_result_t,
    DeviceEpilogOpT,
    InputIteratorTs...>;

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

    if constexpr (can_process_packed_block_path)
    {
      if (full_tile && transform_epilog_packed_path_aligned<pack_size>(tile_item_base, d_out, d_in...))
      {
        transform_epilog_process_packed_block_tile<BlockThreads, ItemsPerThread, pack_size>(
          d_out, transform_op, ::cuda::std::index_sequence_for<InputIteratorTs...>{}, tile_item_base, d_in...);
        continue;
      }
    }

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
