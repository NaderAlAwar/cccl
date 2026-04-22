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

  using input_ref_t                      = cub::detail::it_reference_t<InputIteratorT>;
  constexpr bool transform_accepts_index = ::cuda::std::is_invocable_v<TransformOpT, ::cuda::std::int64_t, input_ref_t>;

  static_assert(transform_accepts_index || ::cuda::std::is_invocable_v<TransformOpT, input_ref_t>,
                "transform_op must be invocable with either (index, value) or (value).");

  using transform_result_t = typename transform_epilog_result<TransformOpT, input_ref_t, transform_accepts_index>::type;

  static_assert(::cuda::std::is_default_constructible_v<transform_result_t>,
                "The current hierarchical transform epilog kernel default-initializes the per-thread transform result "
                "for out-of-range lanes.");

  constexpr int warp_threads    = 32;
  constexpr int warps_per_block = BlockThreads / warp_threads;

  const int lane_rank          = static_cast<int>(threadIdx.x % warp_threads);
  const int warp_rank_in_block = static_cast<int>(threadIdx.x / warp_threads);
  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group             = ::cuda::experimental::this_block{block_hierarchy};

  const auto block_segment_stride = static_cast<::cuda::std::int64_t>(gridDim.x) * warps_per_block;
  const auto block_segment_base   = static_cast<::cuda::std::int64_t>(blockIdx.x) * warps_per_block;
  using block_load_t              = BlockLoad<input_ref_t, BlockThreads, 1, BLOCK_LOAD_DIRECT>;
  __shared__ typename block_load_t::TempStorage temp_storage;

  // Assumption is one item per thread
  __shared__ input_ref_t shared_values[BlockThreads];

  for (::cuda::std::int64_t tile_segment_base = block_segment_base; tile_segment_base < num_segments;
       tile_segment_base += block_segment_stride)
  {
    const auto segment_index = tile_segment_base + warp_rank_in_block; // absolute segment index
    const bool segment_valid = segment_index < num_segments;
    transform_result_t thread_result{};
    input_ref_t loaded_value[1];

    const auto remaining_segments = num_segments - tile_segment_base;
    const int segments_to_load =
      static_cast<int>((::cuda::std::min) (remaining_segments, static_cast<::cuda::std::int64_t>(warps_per_block)));
    const int items_to_load = segments_to_load * segment_size;
    block_load_t(temp_storage)
      .Load(d_in + tile_segment_base * static_cast<::cuda::std::int64_t>(segment_size), loaded_value, items_to_load);

    shared_values[threadIdx.x] = loaded_value[0];
    __syncthreads();

    if (segment_valid && lane_rank < segment_size)
    {
      const auto shared_item_index = lane_rank + warp_rank_in_block * segment_size;
      if constexpr (transform_accepts_index)
      {
        const auto item_index = tile_segment_base * static_cast<::cuda::std::int64_t>(segment_size)
                              + static_cast<::cuda::std::int64_t>(shared_item_index);
        thread_result         = transform_op(item_index, shared_values[shared_item_index]);
        *(d_out + item_index) = thread_result;
      }
      else
      {
        const auto item_index = tile_segment_base * static_cast<::cuda::std::int64_t>(segment_size)
                              + static_cast<::cuda::std::int64_t>(shared_item_index);
        thread_result         = transform_op(shared_values[shared_item_index]);
        *(d_out + item_index) = thread_result;
      }
    }

    device_epilog_op(block_group, segment_valid, segment_index, thread_result);

    __syncthreads();
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
