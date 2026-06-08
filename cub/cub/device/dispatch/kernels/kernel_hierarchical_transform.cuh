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
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/system/cuda/detail/core/util.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <typename DirectInputIteratorT>
struct transform_prolog_direct_vector_load
{
  static constexpr bool supported = false;
};

template <typename T>
struct transform_prolog_direct_vector_load<T*>
{
  using value_t = ::cuda::std::remove_cv_t<T>;

  static constexpr bool supported = transform_prolog_vectorizable_value_v<value_t>;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static bool IsAligned(T* input, int index)
  {
    constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_vector_storage_t));
    return (reinterpret_cast<::cuda::std::uintptr_t>(input + index) % alignment) == 0;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static transform_prolog_vector_storage_t Load(T* input, int index)
  {
    return reinterpret_cast<const transform_prolog_vector_storage_t*>(input + index)[0];
  }
};

template <CacheLoadModifier Modifier, typename T, typename OffsetT>
struct transform_prolog_direct_vector_load<CacheModifiedInputIterator<Modifier, T, OffsetT>>
{
  using value_t = ::cuda::std::remove_cv_t<T>;

  static constexpr bool supported = transform_prolog_vectorizable_value_v<value_t>;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static bool
  IsAligned(CacheModifiedInputIterator<Modifier, T, OffsetT> input, int index)
  {
    constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_vector_storage_t));
    return (reinterpret_cast<::cuda::std::uintptr_t>(input.ptr + index) % alignment) == 0;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static transform_prolog_vector_storage_t
  Load(CacheModifiedInputIterator<Modifier, T, OffsetT> input, int index)
  {
    constexpr int vector_items = transform_prolog_vector_items_v<value_t>;
    CacheModifiedInputIterator<Modifier, transform_prolog_vector_storage_t, OffsetT> vector_input(
      reinterpret_cast<transform_prolog_vector_storage_t*>(input.ptr));
    return vector_input[index / vector_items];
  }
};

template <typename OutputIteratorT>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool
transform_prolog_output_is_vector_aligned(OutputIteratorT output, ::cuda::std::size_t index)
{
  constexpr auto alignment = static_cast<::cuda::std::uintptr_t>(sizeof(transform_prolog_vector_storage_t));
  return (reinterpret_cast<::cuda::std::uintptr_t>(output + index) % alignment) == 0;
}

template <int BlockThreads,
          bool VectorizeOutputs,
          typename ValueT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentResultT,
          typename ApplyElementTransformWithDirectT>
_CCCL_DEVICE _CCCL_FORCEINLINE void transform_prolog_store_outputs(
  const DirectInputIteratorT d_direct,
  const OutputIteratorT d_out,
  ::cuda::std::size_t segment_offset,
  int chunk_begin,
  int items,
  ValueT* shared_segment,
  const SegmentResultT& segment_result,
  ApplyElementTransformWithDirectT apply_element_transform_with_direct)
{
  const int thread_rank = static_cast<int>(threadIdx.x);

  using direct_vector_load_t = transform_prolog_direct_vector_load<::cuda::std::remove_cv_t<DirectInputIteratorT>>;
  using raw_value_t          = ::cuda::std::remove_cv_t<ValueT>;
  using direct_value_t       = ::cuda::std::remove_cv_t<cub::detail::it_value_t<DirectInputIteratorT>>;
  using output_pointer_t     = ::cuda::std::remove_cv_t<OutputIteratorT>;
  using output_value_t       = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<output_pointer_t>>;
  constexpr int vector_items = transform_prolog_vector_items_v<raw_value_t>;
  constexpr bool output_is_contiguous_pointer =
    ::cuda::std::is_pointer_v<::cuda::std::remove_cv_t<OutputIteratorT>>
    && ::cuda::std::is_same_v<output_value_t, raw_value_t>;
  constexpr bool can_vectorize =
    VectorizeOutputs && transform_prolog_vectorizable_value_v<raw_value_t>
    && ::cuda::std::is_same_v<direct_value_t, raw_value_t> && direct_vector_load_t::supported
    && output_is_contiguous_pointer;

  if constexpr (can_vectorize)
  {
    auto store_vectorized_outputs = [&](int vector_count) {
      auto* d_out_vector          = reinterpret_cast<transform_prolog_vector_storage_t*>(d_out + segment_offset);
      auto* shared_segment_vector = reinterpret_cast<const transform_prolog_vector_storage_t*>(shared_segment);

      for (int vector_index = thread_rank; vector_index < vector_count; vector_index += BlockThreads)
      {
        const int local_base       = vector_index * vector_items;
        const int index_in_segment = chunk_begin + local_base;

        using THRUST_NS_QUALIFIER::cuda_cub::core::detail::uninitialized_array;
        uninitialized_array<raw_value_t, vector_items, sizeof(transform_prolog_vector_storage_t)> direct_values;
        uninitialized_array<raw_value_t, vector_items, sizeof(transform_prolog_vector_storage_t)> shared_values;
        reinterpret_cast<transform_prolog_vector_storage_t*>(direct_values.data())[0] =
          direct_vector_load_t::Load(d_direct, index_in_segment);
        reinterpret_cast<transform_prolog_vector_storage_t*>(shared_values.data())[0] =
          shared_segment_vector[local_base / vector_items];

        uninitialized_array<raw_value_t, vector_items, sizeof(transform_prolog_vector_storage_t)> transformed_values;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int lane = 0; lane < vector_items; ++lane)
        {
          transformed_values[lane] =
            apply_element_transform_with_direct(shared_values[lane], direct_values[lane], segment_result);
        }

        d_out_vector[index_in_segment / vector_items] =
          reinterpret_cast<const transform_prolog_vector_storage_t*>(transformed_values.data())[0];
      }
    };

    _CCCL_ASSERT(chunk_begin % vector_items == 0, "");
    _CCCL_ASSERT((segment_offset + static_cast<::cuda::std::size_t>(chunk_begin)) % vector_items == 0, "");
    _CCCL_ASSERT(items % vector_items == 0, "");
    _CCCL_ASSERT(direct_vector_load_t::IsAligned(d_direct, chunk_begin), "");
    _CCCL_ASSERT(
      transform_prolog_output_is_vector_aligned(d_out, segment_offset + static_cast<::cuda::std::size_t>(chunk_begin)),
      "");

    const int vector_count = items / vector_items;
    store_vectorized_outputs(vector_count);
    return;
  }

  for (int local_index = thread_rank; local_index < items; local_index += BlockThreads)
  {
    const int index_in_segment = chunk_begin + local_index;
    const auto direct_value    = d_direct[index_in_segment];
    d_out[segment_offset + static_cast<::cuda::std::size_t>(index_in_segment)] =
      apply_element_transform_with_direct(shared_segment[local_index], direct_value, segment_result);
  }
}

template <int BlockThreads,
          int BulkCopyAlignment,
          bool VectorizeDirectAndOutput,
          typename InputIteratorT,
          typename DirectInputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const DirectInputIteratorT d_direct,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int segment_size,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - the block first stages the segment into shared memory via `BlockLoadToShared`
  // - `segment_op` receives an implementation-chosen per-thread range
  // - `segment_op` is responsible for any block-wide combine it needs and should return the final segment result on
  //   every thread in the block group
  // - the kernel then applies `element_transform_op` to each item, passing the staged input value, the direct input
  //   value, and the segment result

  auto block_hierarchy = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  using block_group_t  = ::cuda::experimental::this_block<decltype(block_hierarchy)>;
  using value_t        = cub::detail::it_value_t<InputIteratorT>;
  using input_range_t  = transform_prolog_segment_range_t<
     BlockThreads,
     block_group_t,
     SegmentOpT,
     value_t,
     transform_prolog_shared_vector_aligned_v<BulkCopyAlignment>,
     true>;
  using segment_result_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, block_group_t, input_range_t>>;
  using block_load_to_shared_t = BlockLoadToShared<BlockThreads>;

  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  extern __shared__ char shared_segment_buffer_base[];
  __shared__ typename block_load_to_shared_t::TempStorage load_to_shared_storage;

  const int segment_id = static_cast<int>(blockIdx.x);
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const auto segment_begin = d_in + segment_offset;

  auto block_group = ::cuda::experimental::this_block{block_hierarchy};
  auto apply_element_transform_with_direct =
    [&](auto&& value, auto&& direct_value, const segment_result_t& segment_result) {
      using input_ref_t  = decltype(value);
      using direct_ref_t = decltype(direct_value);

      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, input_ref_t, direct_ref_t, segment_result_t>,
                    "element_transform_op must be invocable with (value, direct_input, segment_result).");
      return element_transform_op(
        static_cast<input_ref_t>(value), static_cast<direct_ref_t>(direct_value), segment_result);
    };

  const int shared_buffer_bytes = cub::detail::LoadToSharedBufferSizeBytes<value_t, BulkCopyAlignment>(segment_size);

  block_load_to_shared_t load_to_shared{load_to_shared_storage};
  ::cuda::std::span<char> shared_buffer{
    shared_segment_buffer_base, static_cast<::cuda::std::size_t>(shared_buffer_bytes)};
  ::cuda::std::span<const value_t> input_buffer{segment_begin, static_cast<::cuda::std::size_t>(segment_size)};
  auto staged_segment = load_to_shared.template CopyAsync<value_t, BulkCopyAlignment>(shared_buffer, input_buffer);
  auto token          = load_to_shared.Commit();
  load_to_shared.Wait(::cuda::std::move(token));
  value_t* shared_segment = staged_segment.data();

  auto input_range = make_transform_prolog_segment_range<
    BlockThreads,
    block_group_t,
    SegmentOpT,
    value_t,
    transform_prolog_shared_vector_aligned_v<BulkCopyAlignment>,
    true>(shared_segment, segment_size);
  const segment_result_t segment_result = segment_op(block_group, input_range);

  transform_prolog_store_outputs<BlockThreads,
                                 transform_prolog_shared_vector_aligned_v<BulkCopyAlignment> && VectorizeDirectAndOutput>(
    d_direct,
    d_out,
    segment_offset,
    0,
    segment_size,
    shared_segment,
    segment_result,
    apply_element_transform_with_direct);
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
