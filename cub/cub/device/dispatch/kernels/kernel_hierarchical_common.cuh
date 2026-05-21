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

#include <cuda/hierarchy>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/hierarchy.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
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

  template <typename FnT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each(FnT fn) const
  {
    for (int item = 0; item < items_; ++item)
    {
      fn(begin_[item]);
    }
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

template <typename StridedIteratorT>
class striped_thread_segment_range
{
public:
  using iterator       = StridedIteratorT;
  using const_iterator = StridedIteratorT;
  using value_type     = cub::detail::it_value_t<StridedIteratorT>;
  using reference      = cub::detail::it_reference_t<StridedIteratorT>;

  _CCCL_HOST_DEVICE constexpr striped_thread_segment_range() noexcept = default;

  _CCCL_HOST_DEVICE constexpr striped_thread_segment_range(iterator begin, int offset, int items) noexcept
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

  template <typename FnT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each(FnT fn) const
  {
    for (int item = 0; item < items_; ++item)
    {
      fn(begin_[item]);
    }
  }

private:
  iterator begin_{};
  int offset_{0};
  int items_{0};
};

template <int BlockThreads, typename RandomAccessIteratorT>
using striped_thread_segment_iterator_t = decltype(::cuda::make_strided_iterator(
  ::cuda::std::declval<RandomAccessIteratorT>(), ::cuda::std::integral_constant<int, BlockThreads>{}));

template <int BlockThreads, typename RandomAccessIteratorT>
using striped_thread_segment_range_t =
  striped_thread_segment_range<striped_thread_segment_iterator_t<BlockThreads, RandomAccessIteratorT>>;

using transform_prolog_f32_vector_storage_t = int4;
static_assert(sizeof(transform_prolog_f32_vector_storage_t) == 4 * sizeof(float));
static_assert(alignof(transform_prolog_f32_vector_storage_t) == 4 * sizeof(float));

template <int BulkCopyAlignment>
inline constexpr bool transform_prolog_shared_vector_aligned_v =
  BulkCopyAlignment >= static_cast<int>(alignof(transform_prolog_f32_vector_storage_t));

template <int BlockThreads, typename ValueT, bool AssumeFullVectors = false>
class vectorized_thread_segment_range
{
  using raw_value_t = ::cuda::std::remove_cv_t<ValueT>;

public:
  using value_type = ValueT;
  using reference  = value_type;

  class const_iterator
  {
  public:
    _CCCL_DEVICE constexpr const_iterator() noexcept = default;

    _CCCL_DEVICE constexpr const_iterator(const vectorized_thread_segment_range* range, int index) noexcept
        : range_(range)
        , index_(index)
    {}

    _CCCL_DEVICE const_iterator& operator++() noexcept
    {
      ++index_;
      return *this;
    }

    [[nodiscard]] _CCCL_DEVICE reference operator*() const noexcept
    {
      return (*range_)[index_];
    }

    [[nodiscard]] _CCCL_DEVICE bool operator!=(const const_iterator& other) const noexcept
    {
      return index_ != other.index_;
    }

  private:
    const vectorized_thread_segment_range* range_{};
    int index_{0};
  };

  using iterator = const_iterator;

  _CCCL_DEVICE constexpr vectorized_thread_segment_range() noexcept = default;

  _CCCL_DEVICE constexpr vectorized_thread_segment_range(
    ValueT* segment_begin, int segment_size, int first_item, int offset, int items) noexcept
      : segment_begin_(segment_begin)
      , segment_size_(segment_size)
      , first_item_(first_item)
      , offset_(offset)
      , items_(items)
  {}

  [[nodiscard]] _CCCL_DEVICE constexpr iterator begin() const noexcept
  {
    return iterator{this, 0};
  }

  [[nodiscard]] _CCCL_DEVICE constexpr iterator end() const noexcept
  {
    return iterator{this, items_};
  }

  [[nodiscard]] _CCCL_DEVICE constexpr int offset() const noexcept
  {
    return offset_;
  }

  [[nodiscard]] _CCCL_DEVICE constexpr int size() const noexcept
  {
    return items_;
  }

  [[nodiscard]] _CCCL_DEVICE constexpr bool empty() const noexcept
  {
    return items_ == 0;
  }

  _CCCL_DEVICE reference operator[](int idx) const noexcept
  {
    constexpr int vector_items = 4;
    const int vector_group     = idx / vector_items;
    const int vector_lane      = idx % vector_items;
    const int vector_base      = first_item_ + vector_group * BlockThreads * vector_items;
    return segment_begin_[vector_base + vector_lane];
  }

  template <typename FnT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void for_each(FnT fn) const
  {
    constexpr int vector_items = 4;

    if constexpr (AssumeFullVectors)
    {
      _CCCL_ASSERT(segment_size_ % vector_items == 0, "");

      const int thread_rank  = static_cast<int>(threadIdx.x);
      const int vector_count = segment_size_ / vector_items;
      for (int vector_index = thread_rank; vector_index < vector_count; vector_index += BlockThreads)
      {
        const int vector_base = vector_index * vector_items;

        transform_prolog_f32_vector_storage_t values_storage =
          reinterpret_cast<const transform_prolog_f32_vector_storage_t*>(segment_begin_ + vector_base)[0];
        raw_value_t* values = reinterpret_cast<raw_value_t*>(&values_storage);

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int lane = 0; lane < vector_items; ++lane)
        {
          fn(values[lane]);
        }
      }
    }
    else
    {
      for (int vector_group = 0, consumed = 0; consumed < items_; ++vector_group)
      {
        const int vector_base = first_item_ + vector_group * BlockThreads * vector_items;
        const int lane_items  = (::cuda::std::min) (vector_items, items_ - consumed);

        transform_prolog_f32_vector_storage_t values_storage{};
        raw_value_t* values = reinterpret_cast<raw_value_t*>(&values_storage);
        FillValues(values_storage, values, vector_base);

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int lane = 0; lane < vector_items; ++lane)
        {
          if (lane < lane_items)
          {
            fn(values[lane]);
          }
        }

        consumed += lane_items;
      }
    }
  }

private:
  _CCCL_DEVICE void
  FillValues(transform_prolog_f32_vector_storage_t& values_storage, raw_value_t* values, int vector_base) const noexcept
  {
    const bool full_vector = vector_base + 4 <= segment_size_;

    if (full_vector)
    {
      values_storage = reinterpret_cast<const transform_prolog_f32_vector_storage_t*>(segment_begin_ + vector_base)[0];
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int lane = 0; lane < 4; ++lane)
      {
        const int item = vector_base + lane;
        values[lane]   = item < segment_size_ ? segment_begin_[item] : raw_value_t{};
      }
    }
  }

  ValueT* segment_begin_{};
  int segment_size_{0};
  int first_item_{0};
  int offset_{0};
  int items_{0};
};

template <int BlockThreads, bool AssumeFullVectors = false, typename ValueT>
_CCCL_DEVICE auto make_vectorized_thread_segment_range(ValueT* segment_begin, int segment_size, int segment_offset = 0)
{
  constexpr int vector_items = 4;
  const int thread_rank      = static_cast<int>(threadIdx.x);
  const int first_item       = thread_rank * vector_items;

  int items = 0;
  if (first_item < segment_size)
  {
    const int remaining     = segment_size - first_item;
    const int vector_stride = BlockThreads * vector_items;
    items                   = (remaining / vector_stride) * vector_items;
    items += (::cuda::std::min) (vector_items, remaining % vector_stride);
  }

  return vectorized_thread_segment_range<BlockThreads, ValueT, AssumeFullVectors>{
    segment_begin, segment_size, first_item, segment_offset + first_item, items};
}

template <int BlockThreads, typename RandomAccessIteratorT>
_CCCL_DEVICE auto
make_striped_thread_segment_range(RandomAccessIteratorT segment_begin, int segment_size, int segment_offset = 0)
{
  const int thread_rank = static_cast<int>(threadIdx.x);
  const int first_item  = thread_rank < segment_size ? thread_rank : segment_size;
  const int items = thread_rank < segment_size ? ((segment_size - thread_rank + BlockThreads - 1) / BlockThreads) : 0;
  auto begin =
    ::cuda::make_strided_iterator(segment_begin + first_item, ::cuda::std::integral_constant<int, BlockThreads>{});

  return striped_thread_segment_range_t<BlockThreads, RandomAccessIteratorT>{begin, segment_offset + first_item, items};
}

template <int BlockThreads,
          typename GroupT,
          typename SegmentOpT,
          typename ValueT,
          bool SharedSegmentVectorAligned,
          bool AssumeFullVectorizedRange = false>
struct transform_prolog_segment_range_selector
{
  using raw_value_t      = ::cuda::std::remove_cv_t<ValueT>;
  using vectorized_range = vectorized_thread_segment_range<BlockThreads, ValueT, AssumeFullVectorizedRange>;
  using striped_range    = striped_thread_segment_range_t<BlockThreads, ValueT*>;
  using thread_range     = thread_segment_range<ValueT*>;

  // TODO: The vectorized/striped ranges are used so F32 RMSNorm-style reductions read shared memory with consecutive
  // warp lanes touching consecutive 4-byte banks. This is only the right bank-conflict fix for F32/4-byte values;
  // define a dtype-aware policy for smaller/larger value sizes before using these traversals there by default.
  static constexpr bool use_vectorized_range =
    SharedSegmentVectorAligned
    && ::cuda::std::is_same_v<raw_value_t, float> && ::cuda::std::is_invocable_v<SegmentOpT, GroupT, vectorized_range>;
  static constexpr bool use_striped_range =
    !use_vectorized_range && sizeof(raw_value_t) == 4 && ::cuda::std::is_invocable_v<SegmentOpT, GroupT, striped_range>;
  static constexpr bool use_thread_range =
    !use_vectorized_range && !use_striped_range && ::cuda::std::is_invocable_v<SegmentOpT, GroupT, thread_range>;
  static constexpr bool valid = use_vectorized_range || use_striped_range || use_thread_range;

  using type = ::cuda::std::conditional_t<use_vectorized_range,
                                          vectorized_range,
                                          ::cuda::std::conditional_t<use_striped_range, striped_range, thread_range>>;
};

template <int BlockThreads,
          typename GroupT,
          typename SegmentOpT,
          typename ValueT,
          bool SharedSegmentVectorAligned,
          bool AssumeFullVectorizedRange = false>
using transform_prolog_segment_range_t = typename transform_prolog_segment_range_selector<
  BlockThreads,
  GroupT,
  SegmentOpT,
  ValueT,
  SharedSegmentVectorAligned,
  AssumeFullVectorizedRange>::type;

template <int BlockThreads,
          typename GroupT,
          typename SegmentOpT,
          typename ValueT,
          bool SharedSegmentVectorAligned,
          bool AssumeFullVectorizedRange = false>
_CCCL_DEVICE auto make_transform_prolog_segment_range(ValueT* segment_begin, int segment_size, int segment_offset = 0)
{
  using selector_t = transform_prolog_segment_range_selector<
    BlockThreads,
    GroupT,
    SegmentOpT,
    ValueT,
    SharedSegmentVectorAligned,
    AssumeFullVectorizedRange>;
  static_assert(selector_t::valid, "segment_op must be invocable with a TransformProlog segment range.");

  if constexpr (selector_t::use_vectorized_range)
  {
    return make_vectorized_thread_segment_range<BlockThreads, AssumeFullVectorizedRange>(
      segment_begin, segment_size, segment_offset);
  }
  else if constexpr (selector_t::use_striped_range)
  {
    return make_striped_thread_segment_range<BlockThreads>(segment_begin, segment_size, segment_offset);
  }
  else if constexpr (selector_t::use_thread_range)
  {
    const auto local_range = make_thread_segment_range<BlockThreads>(segment_begin, segment_size);
    return thread_segment_range<ValueT*>{local_range.begin(), segment_offset + local_range.offset(), local_range.size()};
  }
  else
  {
    static_assert(selector_t::valid, "segment_op must be invocable with a TransformProlog segment range.");
  }
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
_CCCL_DEVICE T sum(::cuda::experimental::this_block<Hierarchy> group, T thread_data)
{
  return reduce(group, thread_data, ::cuda::std::plus<>{});
}

template <typename Hierarchy, typename T>
_CCCL_DEVICE T sum(::cuda::experimental::this_cluster<Hierarchy> group, T thread_data)
{
  return reduce(group, thread_data, ::cuda::std::plus<>{});
}

template <typename Hierarchy>
_CCCL_DEVICE ::cuda::std::uint32_t ballot(::cuda::experimental::this_warp<Hierarchy> /*group*/, bool predicate)
{
  return static_cast<::cuda::std::uint32_t>(__ballot_sync(0xFFFF'FFFFu, predicate));
}
} // namespace cuda::coop
