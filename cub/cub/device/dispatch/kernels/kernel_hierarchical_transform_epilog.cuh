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
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/__memory/align_down.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
constexpr int transform_epilog_bulk_copy_alignment =
  (CUB_PTX_ARCH >= 900 && CUB_PTX_ARCH < 1000) ? 128 : cub::detail::bulk_copy_min_align;

template <typename T>
struct transform_epilog_aligned_base_ptr
{
  using value_type = ::cuda::std::remove_cv_t<T>;

  const char* ptr;
  int head_padding;
};

template <typename T>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto transform_epilog_make_aligned_base_ptr(T* ptr)
  -> transform_epilog_aligned_base_ptr<T>
{
  const auto raw_ptr  = reinterpret_cast<const char*>(ptr);
  const auto base_ptr = ::cuda::align_down(raw_ptr, transform_epilog_bulk_copy_alignment);
  return {base_ptr, static_cast<int>(raw_ptr - base_ptr)};
}

template <typename T>
struct transform_epilog_is_aligned_base_ptr : ::cuda::std::false_type
{};

template <typename T>
struct transform_epilog_is_aligned_base_ptr<transform_epilog_aligned_base_ptr<T>> : ::cuda::std::true_type
{};

template <typename InputIteratorT>
inline constexpr bool transform_epilog_is_aligned_base_ptr_v =
  transform_epilog_is_aligned_base_ptr<::cuda::std::decay_t<InputIteratorT>>::value;

template <typename InputIteratorT>
struct transform_epilog_input_value
{
  using type = ::cuda::std::remove_cv_t<cub::detail::it_value_t<InputIteratorT>>;
};

template <typename T>
struct transform_epilog_input_value<transform_epilog_aligned_base_ptr<T>>
{
  using type = typename transform_epilog_aligned_base_ptr<T>::value_type;
};

template <typename InputIteratorT>
using transform_epilog_input_value_t =
  typename transform_epilog_input_value<::cuda::std::decay_t<InputIteratorT>>::type;

template <typename InputIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE bool transform_epilog_is_valid_aligned_input(const InputIteratorT&, int)
{
  return true;
}

template <typename T>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE bool
transform_epilog_is_valid_aligned_input(const transform_epilog_aligned_base_ptr<T>& input, int total_items)
{
  const auto total_bytes =
    static_cast<::cuda::std::int64_t>(sizeof(typename transform_epilog_aligned_base_ptr<T>::value_type)) * total_items;
  return input.head_padding == 0 && total_bytes % transform_epilog_bulk_copy_alignment == 0;
}

template <typename TransformOpT, bool AcceptsIndex, typename... InputValueTs>
struct transform_epilog_result
{};

template <typename TransformOpT, typename... InputValueTs>
struct transform_epilog_result<TransformOpT, true, InputValueTs...>
{
  using type = ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, int, InputValueTs...>>;
};

template <typename TransformOpT, typename... InputValueTs>
struct transform_epilog_result<TransformOpT, false, InputValueTs...>
{
  using type = ::cuda::std::decay_t<::cuda::std::invoke_result_t<TransformOpT, InputValueTs...>>;
};

template <int ItemsPerThread, typename InputValueT>
struct transform_epilog_shared_input_values
{
  using value_t = ::cuda::std::remove_cv_t<InputValueT>;

  ::cuda::std::span<value_t> values;

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(::cuda::std::span<value_t> shared_values)
  {
    values = shared_values;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE decltype(auto) get(int item, int linear_tid)
  {
    return values[linear_tid * ItemsPerThread + item];
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE decltype(auto) get_local(int local_item)
  {
    return values[local_item];
  }
};

template <typename InputIteratorT>
inline constexpr bool transform_epilog_use_load_to_shared =
  transform_epilog_is_aligned_base_ptr_v<InputIteratorT>
  && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<
    ::cuda::std::remove_cv_t<transform_epilog_input_value_t<InputIteratorT>>>;

template <typename... InputIteratorTs>
inline constexpr bool transform_epilog_all_load_to_shared =
  (... && transform_epilog_use_load_to_shared<InputIteratorTs>);

_CCCL_HOST_DEVICE constexpr int transform_epilog_align_up(int value, int alignment)
{
  return ((value + alignment - 1) / alignment) * alignment;
}

template <int ItemsPerBlock, typename InputIteratorT>
_CCCL_HOST_DEVICE constexpr int transform_epilog_shared_buffer_align()
{
  return transform_epilog_bulk_copy_alignment;
}

template <int ItemsPerBlock, typename InputIteratorT>
_CCCL_HOST_DEVICE constexpr int transform_epilog_shared_buffer_size()
{
  return static_cast<int>(sizeof(transform_epilog_input_value_t<InputIteratorT>)) * ItemsPerBlock
       + transform_epilog_bulk_copy_alignment;
}

template <int ItemsPerBlock, typename... InputIteratorTs>
_CCCL_HOST_DEVICE constexpr int transform_epilog_shared_buffer_max_align()
{
  int result = 1;
  ((result = (::cuda::std::max) (result, transform_epilog_shared_buffer_align<ItemsPerBlock, InputIteratorTs>())), ...);
  return result;
}

template <int ItemsPerBlock, typename... InputIteratorTs>
_CCCL_HOST_DEVICE constexpr int transform_epilog_shared_buffer_total_size()
{
  int offset = 0;
  ((offset = transform_epilog_align_up(offset, transform_epilog_shared_buffer_align<ItemsPerBlock, InputIteratorTs>()),
    offset += transform_epilog_shared_buffer_size<ItemsPerBlock, InputIteratorTs>()),
   ...);
  return offset;
}

template <int TargetIndex, int ItemsPerBlock, typename... InputIteratorTs>
_CCCL_HOST_DEVICE constexpr int transform_epilog_shared_buffer_offset()
{
  int offset = 0;
  int result = 0;
  int index  = 0;
  ((offset = transform_epilog_align_up(offset, transform_epilog_shared_buffer_align<ItemsPerBlock, InputIteratorTs>()),
    result = index == TargetIndex ? offset : result,
    offset += transform_epilog_shared_buffer_size<ItemsPerBlock, InputIteratorTs>(),
    ++index),
   ...);
  return result;
}

template <int BlockThreads, int ItemsPerThread, typename... InputIteratorTs>
struct transform_epilog_input_load_storage
{
  static constexpr int items_per_block = BlockThreads * ItemsPerThread;
  static constexpr int shared_buffer_align =
    transform_epilog_shared_buffer_max_align<items_per_block, InputIteratorTs...>();
  static constexpr int shared_buffer_bytes =
    transform_epilog_shared_buffer_total_size<items_per_block, InputIteratorTs...>();

  using block_load_to_shared_t = BlockLoadToShared<BlockThreads>;

  typename block_load_to_shared_t::TempStorage load_to_shared;
  alignas(shared_buffer_align) char shared_buffers[shared_buffer_bytes == 0 ? 1 : shared_buffer_bytes];

  template <int Index>
  _CCCL_HOST_DEVICE static constexpr int shared_buffer_offset()
  {
    return transform_epilog_shared_buffer_offset<Index, items_per_block, InputIteratorTs...>();
  }

  template <int Index>
  _CCCL_HOST_DEVICE static constexpr int shared_buffer_size()
  {
    using input_iterator_t = typename ::cuda::std::tuple_element<Index, ::cuda::std::tuple<InputIteratorTs...>>::type;
    return transform_epilog_shared_buffer_size<items_per_block, input_iterator_t>();
  }
};

template <bool TransformAcceptsIndex, typename TransformOpT, typename LoadedInputsT, ::cuda::std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE auto transform_epilog_invoke(
  TransformOpT& transform_op,
  int item_index,
  LoadedInputsT& loaded_inputs,
  int item,
  int linear_tid,
  ::cuda::std::index_sequence<Is...>)
{
  if constexpr (TransformAcceptsIndex)
  {
    return transform_op(item_index, ::cuda::std::get<Is>(loaded_inputs).get(item, linear_tid)...);
  }
  else
  {
    return transform_op(::cuda::std::get<Is>(loaded_inputs).get(item, linear_tid)...);
  }
}

template <bool TransformAcceptsIndex, typename TransformOpT, typename LoadedInputsT, ::cuda::std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE auto transform_epilog_invoke_local(
  TransformOpT& transform_op,
  int item_index,
  LoadedInputsT& loaded_inputs,
  int local_item,
  ::cuda::std::index_sequence<Is...>)
{
  if constexpr (TransformAcceptsIndex)
  {
    return transform_op(item_index, ::cuda::std::get<Is>(loaded_inputs).get_local(local_item)...);
  }
  else
  {
    return transform_op(::cuda::std::get<Is>(loaded_inputs).get_local(local_item)...);
  }
}

template <int ItemsPerThread>
struct transform_epilog_indices_storage
{
  int values[ItemsPerThread];

  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize()
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      values[item] = -1;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(int item, int value)
  {
    values[item] = value;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE auto get() const -> const int (&)[ItemsPerThread]
  {
    return values;
  }
};

struct transform_epilog_no_indices_storage
{
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize() {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set(int, int) {}
};

struct transform_epilog_tile
{
  int item_base;
  int items;
};

template <int BlockThreads, int ItemsPerThread, int StaticSegmentSize>
_CCCL_DEVICE _CCCL_FORCEINLINE transform_epilog_tile transform_epilog_get_tile(int total_items, int segment_size)
{
  constexpr int items_per_block = BlockThreads * ItemsPerThread;

  if constexpr (StaticSegmentSize > 0)
  {
    static_assert(items_per_block % StaticSegmentSize == 0,
                  "Static segment tiles must contain a whole number of segments.");

    const int tile_item_base  = static_cast<int>(blockIdx.x) * items_per_block;
    const int remaining_items = total_items - tile_item_base;
    const int tile_items      = remaining_items < items_per_block ? remaining_items : items_per_block;
    return {tile_item_base, tile_items};
  }
  else
  {
    const int threads_per_segment = segment_size / ItemsPerThread;
    const int segments_per_block  = BlockThreads / threads_per_segment;
    const int total_segments      = total_items / segment_size;
    const int tile_segment_base   = static_cast<int>(blockIdx.x) * segments_per_block;
    const int remaining_segments  = total_segments - tile_segment_base;
    const int tile_segments       = remaining_segments < segments_per_block ? remaining_segments : segments_per_block;
    const int tile_item_base      = tile_segment_base * segment_size;
    const int tile_items          = tile_segments * segment_size;
    return {tile_item_base, tile_items};
  }
}

template <int BlockThreads,
          int ItemsPerThread,
          int StaticSegmentSize,
          bool UseStripedNoEpilog,
          typename OutputIteratorT,
          typename TransformOpT,
          typename DeviceEpilogOpT,
          typename... InputIteratorTs>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformEpilogKernel(
  _CCCL_GRID_CONSTANT const InputIteratorTs... d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int total_items,
  _CCCL_GRID_CONSTANT const int segment_size,
  TransformOpT transform_op,
  DeviceEpilogOpT device_epilog_op)
{
  // Current implementation:
  // - the core transform path is flat and row-oriented
  // - each block covers a whole number of complete segments
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
  static_assert(!UseStripedNoEpilog || transform_epilog_all_load_to_shared<InputIteratorTs...>,
                "The striped no-epilog fast path requires all inputs to use BlockLoadToShared.");
  static_assert(transform_epilog_all_load_to_shared<InputIteratorTs...>,
                "TransformEpilog requires raw pointer inputs with trivially relocatable value types.");

  constexpr bool transform_accepts_index =
    ::cuda::std::is_invocable_v<TransformOpT, int, transform_epilog_input_value_t<InputIteratorTs>...>;

  static_assert(transform_accepts_index
                  || ::cuda::std::is_invocable_v<TransformOpT, transform_epilog_input_value_t<InputIteratorTs>...>,
                "transform_op must be invocable with either (index, values...) or (values...).");

  using transform_result_t =
    typename transform_epilog_result<TransformOpT,
                                     transform_accepts_index,
                                     transform_epilog_input_value_t<InputIteratorTs>...>::type;

  static_assert(::cuda::std::is_default_constructible_v<transform_result_t>,
                "The current hierarchical transform epilog kernel default-initializes the per-thread transform result "
                "for out-of-range lanes.");
  static_assert(::cuda::std::indirectly_writable<OutputIteratorT, const transform_result_t&>,
                "OutputIteratorT must be indirectly writable from the transform result type.");

  const auto block_hierarchy = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group           = ::cuda::experimental::this_block{block_hierarchy};
  using block_group_t        = decltype(block_group);
  using results_arg_t        = const transform_result_t(&)[ItemsPerThread];
  using indices_arg_t        = const int (&)[ItemsPerThread];
  constexpr bool epilog_accepts_indices =
    ::cuda::std::is_invocable_v<DeviceEpilogOpT, block_group_t, results_arg_t, indices_arg_t>;

  static_assert(epilog_accepts_indices || ::cuda::std::is_invocable_v<DeviceEpilogOpT, block_group_t, results_arg_t>,
                "device_epilog_op must be invocable with either "
                "(block_group, results) or (block_group, results, indices).");

  using indices_storage_t =
    ::cuda::std::conditional_t<epilog_accepts_indices,
                               transform_epilog_indices_storage<ItemsPerThread>,
                               transform_epilog_no_indices_storage>;
  using input_load_storage_t = transform_epilog_input_load_storage<BlockThreads, ItemsPerThread, InputIteratorTs...>;
  using loaded_inputs_t      = ::cuda::std::tuple<
         transform_epilog_shared_input_values<ItemsPerThread, transform_epilog_input_value_t<InputIteratorTs>>...>;

  constexpr int max_items_per_block = BlockThreads * ItemsPerThread;
  const auto linear_tid             = static_cast<int>(threadIdx.x);
  __shared__ input_load_storage_t temp_storage;

  loaded_inputs_t loaded_inputs;
  const auto tile =
    transform_epilog_get_tile<BlockThreads, ItemsPerThread, StaticSegmentSize>(total_items, segment_size);
  const int tile_item_base  = tile.item_base;
  const int tile_items      = tile.items;
  const int tile_item_limit = tile_item_base + tile_items;

  typename input_load_storage_t::block_load_to_shared_t load_to_shared{temp_storage.load_to_shared};

  auto issue_load_to_shared = [&]<::cuda::std::size_t InputIndex, typename InputIteratorT>(
                                ::cuda::std::integral_constant<::cuda::std::size_t, InputIndex>, InputIteratorT input) {
    static_assert(transform_epilog_use_load_to_shared<InputIteratorT>);

    using input_value_t         = transform_epilog_input_value_t<InputIteratorT>;
    constexpr int gmem_align    = transform_epilog_bulk_copy_alignment;
    constexpr int buffer_offset = input_load_storage_t::template shared_buffer_offset<static_cast<int>(InputIndex)>();
    constexpr int buffer_size   = input_load_storage_t::template shared_buffer_size<static_cast<int>(InputIndex)>();

    const int bytes_to_copy    = static_cast<int>(sizeof(input_value_t)) * tile_items;
    const char* const gmem_src = input.ptr + static_cast<::cuda::std::size_t>(tile_item_base) * sizeof(input_value_t);

    _CCCL_ASSERT(input.head_padding == 0, "TransformEpilog raw inputs must satisfy the bulk copy alignment.");
    _CCCL_ASSERT(bytes_to_copy % gmem_align == 0,
                 "TransformEpilog raw input tiles must end on the bulk copy alignment.");
    _CCCL_ASSERT(::cuda::is_aligned(temp_storage.shared_buffers + buffer_offset, gmem_align),
                 "TransformEpilog raw shared tiles must satisfy the bulk copy alignment.");

    ::cuda::std::span<char> shared_buffer{temp_storage.shared_buffers + buffer_offset, buffer_size};
    ::cuda::std::span<const char> gmem_span{gmem_src, static_cast<::cuda::std::size_t>(bytes_to_copy)};
    load_to_shared.template CopyAsync<char, transform_epilog_bulk_copy_alignment>(shared_buffer, gmem_span);
  };

  [&]<::cuda::std::size_t... Is>(::cuda::std::index_sequence<Is...>, InputIteratorTs... input_iterators) {
    (issue_load_to_shared(::cuda::std::integral_constant<::cuda::std::size_t, Is>{}, input_iterators), ...);
  }(::cuda::std::index_sequence_for<InputIteratorTs...>{}, d_in...);

  auto token = load_to_shared.Commit();

  using input_iterators_tuple_t = ::cuda::std::tuple<InputIteratorTs...>;
  auto bind_shared_input =
    [&]<::cuda::std::size_t InputIndex>(::cuda::std::integral_constant<::cuda::std::size_t, InputIndex>) {
      using input_iterator_t = typename ::cuda::std::tuple_element<InputIndex, input_iterators_tuple_t>::type;
      static_assert(transform_epilog_use_load_to_shared<input_iterator_t>);

      using input_value_t         = transform_epilog_input_value_t<input_iterator_t>;
      constexpr int buffer_offset = input_load_storage_t::template shared_buffer_offset<static_cast<int>(InputIndex)>();
      auto* shared_values         = ::cuda::ptr_rebind<input_value_t>(temp_storage.shared_buffers + buffer_offset);

      ::cuda::std::get<InputIndex>(loaded_inputs).set({shared_values, static_cast<::cuda::std::size_t>(tile_items)});
    };

  [&]<::cuda::std::size_t... Is>(::cuda::std::index_sequence<Is...>) {
    (bind_shared_input(::cuda::std::integral_constant<::cuda::std::size_t, Is>{}), ...);
  }(::cuda::std::index_sequence_for<InputIteratorTs...>{});

  load_to_shared.Wait(::cuda::std::move(token));

  if constexpr (UseStripedNoEpilog)
  {
    auto process_tile = [&](auto full_tile) {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const int local_item = item * BlockThreads + linear_tid;
        if (full_tile || local_item < tile_items)
        {
          const int item_index  = tile_item_base + local_item;
          *(d_out + item_index) = transform_epilog_invoke_local<transform_accepts_index>(
            transform_op, item_index, loaded_inputs, local_item, ::cuda::std::index_sequence_for<InputIteratorTs...>{});
        }
      }
    };

    if (tile_items == max_items_per_block)
    {
      process_tile(::cuda::std::true_type{});
    }
    else
    {
      process_tile(::cuda::std::false_type{});
    }

    return;
  }

  transform_result_t output_values[ItemsPerThread]{};
  indices_storage_t indices_storage;
  const int thread_item_base = tile_item_base + linear_tid * ItemsPerThread;

  indices_storage.initialize();

  if (thread_item_base + ItemsPerThread <= tile_item_limit)
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int item_index = thread_item_base + item;

      output_values[item] = transform_epilog_invoke<transform_accepts_index>(
        transform_op,
        item_index,
        loaded_inputs,
        item,
        linear_tid,
        ::cuda::std::index_sequence_for<InputIteratorTs...>{});

      indices_storage.set(item, item_index);
    }
  }
  else
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int item_index = thread_item_base + item;
      if (item_index >= tile_item_limit)
      {
        continue;
      }

      output_values[item] = transform_epilog_invoke<transform_accepts_index>(
        transform_op,
        item_index,
        loaded_inputs,
        item,
        linear_tid,
        ::cuda::std::index_sequence_for<InputIteratorTs...>{});

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

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const int item_index = thread_item_base + item;
    if (item_index < tile_item_limit)
    {
      *(d_out + item_index) = output_values[item];
    }
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
