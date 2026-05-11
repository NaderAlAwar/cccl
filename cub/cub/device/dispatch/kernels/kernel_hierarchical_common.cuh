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

#ifndef _CUDAX_HIERARCHY
#  define _CUDAX_HIERARCHY
#endif

#include <cub/block/block_reduce.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/hierarchy>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/hierarchy.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
template <typename T>
inline constexpr bool hierarchical_transform_stageable_value_v =
  THRUST_NS_QUALIFIER::is_trivially_relocatable_v<::cuda::std::remove_cv_t<T>>;

template <typename InputIteratorT>
inline constexpr bool hierarchical_transform_stageable_input_v =
  hierarchical_transform_stageable_value_v<cub::detail::it_value_t<InputIteratorT>>;

template <typename RandomAccessIteratorT>
class thread_segment_range
{
public:
  using iterator       = RandomAccessIteratorT;
  using const_iterator = RandomAccessIteratorT;
  using value_type     = cub::detail::it_value_t<RandomAccessIteratorT>;
  using reference      = cub::detail::it_reference_t<RandomAccessIteratorT>;

  _CCCL_HOST_DEVICE constexpr thread_segment_range() noexcept = default;

  _CCCL_HOST_DEVICE constexpr thread_segment_range(iterator begin, int offset, int items) noexcept
      : begin_(begin)
      , offset_(offset)
      , items_(items)
  {}

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr iterator begin() const noexcept
  {
    return begin_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr iterator end() const noexcept
  {
    return begin_ + items_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int offset() const noexcept
  {
    return offset_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int size() const noexcept
  {
    return items_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr bool empty() const noexcept
  {
    return items_ == 0;
  }

  _CCCL_HOST_DEVICE constexpr reference operator[](int idx) const noexcept
  {
    return begin_[idx];
  }

private:
  iterator begin_{};
  int offset_{0};
  int items_{0};
};

template <int BlockThreads, typename RandomAccessIteratorT>
_CCCL_DEVICE auto make_thread_segment_range(RandomAccessIteratorT segment_begin, int segment_size)
{
  const int thread_rank   = static_cast<int>(threadIdx.x);
  const int base_items    = segment_size / BlockThreads;
  const int remainder     = segment_size % BlockThreads;
  const int thread_items  = base_items + (thread_rank < remainder ? 1 : 0);
  const int thread_offset = thread_rank * base_items + ((thread_rank < remainder) ? thread_rank : remainder);

  return thread_segment_range<RandomAccessIteratorT>{segment_begin + thread_offset, thread_offset, thread_items};
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END

namespace cuda::coop
{
template <typename Hierarchy, typename T, typename ReductionOp>
_CCCL_DEVICE T reduce(::cuda::experimental::this_block<Hierarchy> group, T thread_data, ReductionOp reduction_op)
{
  using block_extents_t = decltype(::cuda::gpu_thread.extents(::cuda::block, ::cuda::std::declval<Hierarchy>()));

  static_assert(block_extents_t::rank_dynamic() == 0,
                "cuda::coop::reduce currently requires statically sized block groups.");
  static_assert(::cuda::std::is_trivially_copyable_v<T>,
                "cuda::coop::reduce currently requires trivially copyable values.");

  using collective_t =
    cub::BlockReduce<T,
                     static_cast<int>(block_extents_t::static_extent(0)),
                     cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                     static_cast<int>(block_extents_t::static_extent(1)),
                     static_cast<int>(block_extents_t::static_extent(2))>;

  __shared__ T block_result;

  const T thread_result = collective_t{}.Reduce(thread_data, reduction_op);

  if (::cuda::gpu_thread.is_root_rank(group))
  {
    block_result = thread_result;
  }

  group.sync();

  return block_result;
}

template <typename Hierarchy, typename T, typename ReductionOp>
_CCCL_DEVICE T reduce(::cuda::experimental::this_cluster<Hierarchy> group, T thread_data, ReductionOp reduction_op)
{
  using block_extents_t = decltype(::cuda::gpu_thread.extents(::cuda::block, ::cuda::std::declval<Hierarchy>()));

  static_assert(block_extents_t::rank_dynamic() == 0,
                "cuda::coop::reduce currently requires statically sized block groups.");
  static_assert(::cuda::std::is_trivially_copyable_v<T>,
                "cuda::coop::reduce currently requires trivially copyable values.");

  using collective_t =
    cub::BlockReduce<T,
                     static_cast<int>(block_extents_t::static_extent(0)),
                     cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                     static_cast<int>(block_extents_t::static_extent(1)),
                     static_cast<int>(block_extents_t::static_extent(2))>;

  union shared_storage_t
  {
    typename collective_t::TempStorage block_reduce;
    T cluster_partial;
  };

  __shared__ shared_storage_t shared_storage;

  const T block_result = collective_t{shared_storage.block_reduce}.Reduce(thread_data, reduction_op);
  __syncthreads();

  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    ({
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
      {
        shared_storage.cluster_partial = block_result;
      }

      group.sync_aligned();

      if (::cuda::gpu_thread.is_root_rank(group))
      {
        const auto cluster_blocks = static_cast<unsigned int>(::cuda::block.count(group));
        auto root_partial         = static_cast<T*>(__cluster_map_shared_rank(&shared_storage.cluster_partial, 0));
        T cluster_result          = *root_partial;

        for (unsigned int rank = 1; rank < cluster_blocks; ++rank)
        {
          auto remote_partial = static_cast<T*>(__cluster_map_shared_rank(&shared_storage.cluster_partial, rank));
          cluster_result      = reduction_op(cluster_result, *remote_partial);
        }

        for (unsigned int rank = 0; rank < cluster_blocks; ++rank)
        {
          auto remote_partial = static_cast<T*>(__cluster_map_shared_rank(&shared_storage.cluster_partial, rank));
          *remote_partial     = cluster_result;
        }
      }

      group.sync_aligned();
      return shared_storage.cluster_partial;
    }),
    (return T{};))
}

template <typename Hierarchy, typename T>
_CCCL_DEVICE T reduce(::cuda::experimental::this_block<Hierarchy> group, T thread_data)
{
  return reduce(group, thread_data, ::cuda::std::plus<>{});
}

template <typename Hierarchy, typename T>
_CCCL_DEVICE T reduce(::cuda::experimental::this_cluster<Hierarchy> group, T thread_data)
{
  return reduce(group, thread_data, ::cuda::std::plus<>{});
}

template <typename Hierarchy>
_CCCL_DEVICE ::cuda::std::uint32_t ballot(::cuda::experimental::this_warp<Hierarchy> /*group*/, bool predicate)
{
  return static_cast<::cuda::std::uint32_t>(__ballot_sync(0xFFFF'FFFFu, predicate));
}
} // namespace cuda::coop
