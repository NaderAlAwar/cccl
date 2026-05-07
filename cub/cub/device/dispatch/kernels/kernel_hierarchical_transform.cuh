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
#include <cub/block/block_load_to_shared.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/kernels/kernel_hierarchical_common.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/round_up.h>
#include <cuda/__memory/align_down.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::hierarchical
{
constexpr int transform_prolog_bulk_copy_alignment =
  (CUB_PTX_ARCH >= 900 && CUB_PTX_ARCH < 1000) ? 128 : cub::detail::bulk_copy_min_align;

_CCCL_HOST_DEVICE constexpr int transform_prolog_bulk_copy_alignment_for_ptx(int ptx_version)
{
  return (ptx_version >= 900 && ptx_version < 1000) ? 128 : cub::detail::bulk_copy_min_align;
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_alignment(int bulk_copy_alignment)
{
  constexpr int buffer_alignment = cub::detail::LoadToSharedBufferAlignBytes<T>();
  return bulk_copy_alignment > buffer_alignment ? bulk_copy_alignment : buffer_alignment;
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_alignment()
{
  return transform_prolog_load_to_shared_buffer_alignment<T>(transform_prolog_bulk_copy_alignment);
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_size(int items, int bulk_copy_alignment)
{
  if (items == 0)
  {
    return 0;
  }

  const auto payload_bytes   = static_cast<::cuda::std::size_t>(items) * static_cast<::cuda::std::size_t>(sizeof(T));
  const int max_head_padding = bulk_copy_alignment - 1;
  return cub::detail::LoadToSharedBufferSizeBytes<char>(payload_bytes + max_head_padding);
}

template <typename T>
_CCCL_HOST_DEVICE constexpr int transform_prolog_load_to_shared_buffer_size(int items)
{
  return transform_prolog_load_to_shared_buffer_size<T>(items, transform_prolog_bulk_copy_alignment);
}

_CCCL_HOST_DEVICE constexpr ::cuda::std::size_t transform_prolog_align_up(::cuda::std::size_t value, int alignment)
{
  return ((value + static_cast<::cuda::std::size_t>(alignment) - 1) / static_cast<::cuda::std::size_t>(alignment))
       * static_cast<::cuda::std::size_t>(alignment);
}

_CCCL_DEVICE _CCCL_FORCEINLINE const char*
transform_prolog_align_copy_source(const char* source, ::cuda::std::size_t bytes_before_source, int& head_padding)
{
  const char* const aligned_source = ::cuda::align_down(source, transform_prolog_bulk_copy_alignment);
  head_padding                     = static_cast<int>(source - aligned_source);
  if (static_cast<::cuda::std::size_t>(head_padding) <= bytes_before_source)
  {
    return aligned_source;
  }

  head_padding = 0;
  return source;
}

template <int Alignment>
_CCCL_DEVICE _CCCL_FORCEINLINE char* align_dynamic_shared_buffer(char* shared_buffer_base)
{
  if constexpr (Alignment > 16)
  {
    uint32_t shared_buffer_ptr = __cvta_generic_to_shared(shared_buffer_base);
    shared_buffer_ptr          = ::cuda::round_up(shared_buffer_ptr, static_cast<uint32_t>(Alignment));
    asm("" : "+r"(shared_buffer_ptr));
    return static_cast<char*>(__cvta_shared_to_generic(shared_buffer_ptr));
  }
  else
  {
    return shared_buffer_base;
  }
}

struct transform_prolog_identity_source_offset
{
  template <::cuda::std::size_t SourceIndex>
  _CCCL_HOST_DEVICE constexpr ::cuda::std::size_t
  operator()(::cuda::std::integral_constant<::cuda::std::size_t, SourceIndex>,
             ::cuda::std::size_t absolute_logical_offset,
             int) const
  {
    return absolute_logical_offset;
  }
};

struct transform_prolog_first_staged_source
{
  template <typename LogicalIteratorT, typename StagedSourcesT>
  _CCCL_HOST_DEVICE constexpr auto operator()(LogicalIteratorT, StagedSourcesT staged_sources, ::cuda::std::size_t) const
    -> ::cuda::std::tuple_element_t<0, StagedSourcesT>
  {
    return ::cuda::std::get<0>(staged_sources);
  }
};

template <typename LogicalIteratorT,
          typename SourceIteratorTupleT,
          typename BindStagedSourcesOpT,
          typename SourceOffsetOpT = transform_prolog_identity_source_offset>
class transform_prolog_stageable_iterator
{
public:
  using logical_iterator_t       = LogicalIteratorT;
  using source_iterator_tuple_t  = SourceIteratorTupleT;
  using bind_staged_sources_op_t = BindStagedSourcesOpT;
  using source_offset_op_t       = SourceOffsetOpT;

  using value_type        = cub::detail::it_value_t<logical_iterator_t>;
  using difference_type   = cub::detail::it_difference_t<logical_iterator_t>;
  using pointer           = cub::detail::it_pointer_t<logical_iterator_t>;
  using reference         = cub::detail::it_reference_t<logical_iterator_t>;
  using iterator_category = typename ::cuda::std::iterator_traits<logical_iterator_t>::iterator_category;
  using iterator_concept  = iterator_category;

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator() = default;

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator(
    logical_iterator_t logical_iterator,
    source_iterator_tuple_t source_iterators,
    bind_staged_sources_op_t bind_staged_sources_op,
    source_offset_op_t source_offset_op = {})
      : logical_iterator_(::cuda::std::move(logical_iterator))
      , source_iterators_(::cuda::std::move(source_iterators))
      , bind_staged_sources_op_(::cuda::std::move(bind_staged_sources_op))
      , source_offset_op_(::cuda::std::move(source_offset_op))
  {}

  _CCCL_HOST_DEVICE constexpr reference operator*() const
  {
    return *logical_iterator_;
  }

  _CCCL_HOST_DEVICE constexpr reference operator[](difference_type offset) const
  {
    return logical_iterator_[offset];
  }

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator& operator+=(difference_type offset)
  {
    logical_iterator_ += offset;
    logical_base_offset_ =
      static_cast<::cuda::std::size_t>(static_cast<difference_type>(logical_base_offset_) + offset);
    return *this;
  }

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator& operator-=(difference_type offset)
  {
    return *this += -offset;
  }

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator& operator++()
  {
    return *this += 1;
  }

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator operator++(int)
  {
    auto old = *this;
    ++*this;
    return old;
  }

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator& operator--()
  {
    return *this -= 1;
  }

  _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator operator--(int)
  {
    auto old = *this;
    --*this;
    return old;
  }

  friend _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator
  operator+(transform_prolog_stageable_iterator it, difference_type offset)
  {
    it += offset;
    return it;
  }

  friend _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator
  operator+(difference_type offset, transform_prolog_stageable_iterator it)
  {
    it += offset;
    return it;
  }

  friend _CCCL_HOST_DEVICE constexpr transform_prolog_stageable_iterator
  operator-(transform_prolog_stageable_iterator it, difference_type offset)
  {
    it -= offset;
    return it;
  }

  friend _CCCL_HOST_DEVICE constexpr difference_type
  operator-(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return lhs.logical_iterator_ - rhs.logical_iterator_;
  }

  friend _CCCL_HOST_DEVICE constexpr bool
  operator==(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return lhs.logical_iterator_ == rhs.logical_iterator_;
  }

  friend _CCCL_HOST_DEVICE constexpr bool
  operator!=(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return !(lhs == rhs);
  }

  friend _CCCL_HOST_DEVICE constexpr bool
  operator<(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return lhs.logical_iterator_ < rhs.logical_iterator_;
  }

  friend _CCCL_HOST_DEVICE constexpr bool
  operator>(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return rhs < lhs;
  }

  friend _CCCL_HOST_DEVICE constexpr bool
  operator<=(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return !(rhs < lhs);
  }

  friend _CCCL_HOST_DEVICE constexpr bool
  operator>=(const transform_prolog_stageable_iterator& lhs, const transform_prolog_stageable_iterator& rhs)
  {
    return !(lhs < rhs);
  }

  _CCCL_HOST_DEVICE constexpr const source_iterator_tuple_t& source_iterators() const
  {
    return source_iterators_;
  }

  _CCCL_HOST_DEVICE constexpr ::cuda::std::size_t logical_base_offset() const
  {
    return logical_base_offset_;
  }

  template <::cuda::std::size_t SourceIndex>
  _CCCL_HOST_DEVICE constexpr ::cuda::std::size_t
  source_offset(::cuda::std::integral_constant<::cuda::std::size_t, SourceIndex> source_index,
                ::cuda::std::size_t absolute_logical_offset,
                int items) const
  {
    return source_offset_op_(source_index, absolute_logical_offset, items);
  }

  template <typename StagedSourcesT>
  _CCCL_DEVICE constexpr auto
  bind_staged_sources(StagedSourcesT staged_sources, ::cuda::std::size_t absolute_logical_offset) const
  {
    return bind_staged_sources_op_(logical_iterator_, ::cuda::std::move(staged_sources), absolute_logical_offset);
  }

private:
  logical_iterator_t logical_iterator_{};
  source_iterator_tuple_t source_iterators_{};
  bind_staged_sources_op_t bind_staged_sources_op_{};
  source_offset_op_t source_offset_op_{};
  ::cuda::std::size_t logical_base_offset_{0};
};

template <typename LogicalIteratorT,
          typename SourceIteratorTupleT,
          typename BindStagedSourcesOpT,
          typename SourceOffsetOpT = transform_prolog_identity_source_offset>
_CCCL_HOST_DEVICE constexpr auto make_transform_prolog_stageable_iterator(
  LogicalIteratorT logical_iterator,
  SourceIteratorTupleT source_iterators,
  BindStagedSourcesOpT bind_staged_sources_op,
  SourceOffsetOpT source_offset_op = {})
{
  return transform_prolog_stageable_iterator<::cuda::std::decay_t<LogicalIteratorT>,
                                             ::cuda::std::decay_t<SourceIteratorTupleT>,
                                             ::cuda::std::decay_t<BindStagedSourcesOpT>,
                                             ::cuda::std::decay_t<SourceOffsetOpT>>{
    ::cuda::std::move(logical_iterator),
    ::cuda::std::move(source_iterators),
    ::cuda::std::move(bind_staged_sources_op),
    ::cuda::std::move(source_offset_op)};
}

template <typename IteratorT, typename = void>
struct transform_prolog_staging_traits
{
  using staged_iterator_t = IteratorT;
  struct staging_state_t
  {};

  static constexpr bool is_supported = false;
  static constexpr int source_count  = 0;

  _CCCL_HOST_DEVICE static constexpr int shared_buffer_alignment(int)
  {
    return 1;
  }

  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t shared_buffer_size(int, int)
  {
    return 0;
  }

  template <typename IssueCopyAsyncT>
  _CCCL_DEVICE static void issue_load_to_shared(
    IteratorT, ::cuda::std::size_t, int, char*, ::cuda::std::size_t&, staging_state_t&, IssueCopyAsyncT&&)
  {}

  _CCCL_DEVICE static staged_iterator_t bind_shared_input(IteratorT it, const staging_state_t&)
  {
    return it;
  }
};

template <typename IteratorT>
using transform_prolog_staging_traits_t = transform_prolog_staging_traits<::cuda::std::remove_cv_t<IteratorT>>;

template <typename IteratorT>
inline constexpr bool transform_prolog_stageable_input_v = transform_prolog_staging_traits_t<IteratorT>::is_supported;

template <typename IteratorT>
using transform_prolog_staged_iterator_t = typename transform_prolog_staging_traits_t<IteratorT>::staged_iterator_t;

template <typename... IteratorTs>
_CCCL_HOST_DEVICE constexpr int transform_prolog_staging_tuple_alignment(int bulk_copy_alignment)
{
  int result = 1;
  ((result =
      (::cuda::std::max) (result,
                          transform_prolog_staging_traits_t<IteratorTs>::shared_buffer_alignment(bulk_copy_alignment))),
   ...);
  return result;
}

template <typename... IteratorTs>
_CCCL_HOST_DEVICE constexpr ::cuda::std::size_t transform_prolog_staging_tuple_size(int items, int bulk_copy_alignment)
{
  ::cuda::std::size_t offset = 0;
  ((offset = transform_prolog_align_up(
      offset, transform_prolog_staging_traits_t<IteratorTs>::shared_buffer_alignment(bulk_copy_alignment)),
    offset += transform_prolog_staging_traits_t<IteratorTs>::shared_buffer_size(items, bulk_copy_alignment)),
   ...);
  return offset;
}

template <typename IteratorT>
struct transform_prolog_segment_iterator
{
  using type = IteratorT;
};

template <typename IteratorTupleT>
struct transform_prolog_segment_iterator<THRUST_NS_QUALIFIER::zip_iterator<IteratorTupleT>>
{
  using type = ::cuda::std::tuple_element_t<0, IteratorTupleT>;
};

template <typename IteratorT>
using transform_prolog_segment_iterator_t =
  typename transform_prolog_segment_iterator<::cuda::std::remove_cv_t<IteratorT>>::type;

template <typename IteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE IteratorT transform_prolog_segment_input(IteratorT it)
{
  return it;
}

template <typename IteratorTupleT>
_CCCL_DEVICE _CCCL_FORCEINLINE auto transform_prolog_segment_input(THRUST_NS_QUALIFIER::zip_iterator<IteratorTupleT> it)
{
  return ::cuda::std::get<0>(it.get_iterator_tuple());
}

template <typename IteratorT>
struct transform_prolog_staging_traits<IteratorT,
                                       ::cuda::std::enable_if_t<THRUST_NS_QUALIFIER::is_contiguous_iterator_v<IteratorT>>>
{
  using source_value_t    = ::cuda::std::remove_cv_t<cub::detail::it_value_t<IteratorT>>;
  using staged_iterator_t = source_value_t*;
  struct staging_state_t
  {
    char* shared_buffer{nullptr};
    int source_head_padding{0};
  };

  static constexpr bool is_supported = hierarchical_transform_stageable_value_v<source_value_t>;
  static constexpr int source_count  = 1;

  _CCCL_HOST_DEVICE static constexpr int shared_buffer_alignment(int bulk_copy_alignment)
  {
    return transform_prolog_load_to_shared_buffer_alignment<source_value_t>(bulk_copy_alignment);
  }

  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t shared_buffer_size(int items, int bulk_copy_alignment)
  {
    return static_cast<::cuda::std::size_t>(
      transform_prolog_load_to_shared_buffer_size<source_value_t>(items, bulk_copy_alignment));
  }

  template <typename IssueCopyAsyncT>
  _CCCL_DEVICE static void issue_load_to_shared(
    IteratorT it,
    ::cuda::std::size_t logical_offset,
    int items,
    char* shared_buffer_base,
    ::cuda::std::size_t& shared_buffer_offset,
    staging_state_t& staging_state,
    IssueCopyAsyncT&& issue_copy_async)
  {
    constexpr int source_alignment = transform_prolog_load_to_shared_buffer_alignment<source_value_t>();
    shared_buffer_offset           = transform_prolog_align_up(shared_buffer_offset, source_alignment);

    char* source_shared_buffer        = shared_buffer_base + shared_buffer_offset;
    const int payload_bytes           = items * static_cast<int>(sizeof(source_value_t));
    staging_state.shared_buffer       = source_shared_buffer;
    staging_state.source_head_padding = 0;

    if (items <= 0)
    {
      return;
    }

    auto source_it             = it + static_cast<cub::detail::it_difference_t<IteratorT>>(logical_offset);
    const auto source_ptr      = THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(source_it);
    const char* const source   = reinterpret_cast<const char*>(source_ptr);
    int source_head_padding    = 0;
    const char* aligned_source = transform_prolog_align_copy_source(
      source, logical_offset * static_cast<::cuda::std::size_t>(sizeof(source_value_t)), source_head_padding);
    const int bytes_to_copy           = source_head_padding + payload_bytes;
    const int shared_bytes            = cub::detail::LoadToSharedBufferSizeBytes<char>(bytes_to_copy);
    staging_state.source_head_padding = source_head_padding;
    shared_buffer_offset += static_cast<::cuda::std::size_t>(shared_bytes);

    ::cuda::std::span<char> shared_buffer{source_shared_buffer, static_cast<::cuda::std::size_t>(shared_bytes)};
    ::cuda::std::span<const char> input_buffer{aligned_source, static_cast<::cuda::std::size_t>(bytes_to_copy)};
    issue_copy_async(shared_buffer, input_buffer);
  }

  _CCCL_DEVICE static staged_iterator_t bind_shared_input(IteratorT, const staging_state_t& staging_state)
  {
    return reinterpret_cast<source_value_t*>(staging_state.shared_buffer + staging_state.source_head_padding);
  }
};

template <typename IteratorTupleT>
struct transform_prolog_zip_staging_traits;

template <typename... IteratorTs>
struct transform_prolog_zip_staging_traits<::cuda::std::tuple<IteratorTs...>>
{
  using staged_iterator_t =
    THRUST_NS_QUALIFIER::zip_iterator<::cuda::std::tuple<transform_prolog_staged_iterator_t<IteratorTs>...>>;
  using staging_state_t =
    ::cuda::std::tuple<typename transform_prolog_staging_traits_t<IteratorTs>::staging_state_t...>;

  static constexpr bool is_supported = (... && transform_prolog_stageable_input_v<IteratorTs>);
  static constexpr int source_count  = (transform_prolog_staging_traits_t<IteratorTs>::source_count + ... + 0);

  _CCCL_HOST_DEVICE static constexpr int shared_buffer_alignment(int bulk_copy_alignment)
  {
    return transform_prolog_staging_tuple_alignment<IteratorTs...>(bulk_copy_alignment);
  }

  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t shared_buffer_size(int items, int bulk_copy_alignment)
  {
    return transform_prolog_staging_tuple_size<IteratorTs...>(items, bulk_copy_alignment);
  }

  template <typename IssueCopyAsyncT, ::cuda::std::size_t... Is>
  _CCCL_DEVICE static void issue_load_to_shared_impl(
    const ::cuda::std::tuple<IteratorTs...>& iterators,
    ::cuda::std::size_t logical_offset,
    int items,
    char* shared_buffer_base,
    ::cuda::std::size_t& shared_buffer_offset,
    staging_state_t& staging_state,
    IssueCopyAsyncT&& issue_copy_async,
    ::cuda::std::index_sequence<Is...>)
  {
    (
      [&] {
        using child_iterator_t = ::cuda::std::tuple_element_t<Is, ::cuda::std::tuple<IteratorTs...>>;
        using child_traits_t   = transform_prolog_staging_traits_t<child_iterator_t>;

        child_traits_t::issue_load_to_shared(
          ::cuda::std::get<Is>(iterators),
          logical_offset,
          items,
          shared_buffer_base,
          shared_buffer_offset,
          ::cuda::std::get<Is>(staging_state),
          issue_copy_async);
      }(),
      ...);
  }

  template <::cuda::std::size_t... Is>
  _CCCL_DEVICE static staged_iterator_t bind_shared_input_impl(
    const ::cuda::std::tuple<IteratorTs...>& iterators,
    const staging_state_t& staging_state,
    ::cuda::std::index_sequence<Is...>)
  {
    return staged_iterator_t{::cuda::std::make_tuple(([&] {
      using child_iterator_t = ::cuda::std::tuple_element_t<Is, ::cuda::std::tuple<IteratorTs...>>;
      using child_traits_t   = transform_prolog_staging_traits_t<child_iterator_t>;

      return child_traits_t::bind_shared_input(::cuda::std::get<Is>(iterators), ::cuda::std::get<Is>(staging_state));
    }())...)};
  }
};

template <typename LogicalIteratorT, typename... SourceIteratorTs, typename BindStagedSourcesOpT, typename SourceOffsetOpT>
struct transform_prolog_staging_traits<transform_prolog_stageable_iterator<LogicalIteratorT,
                                                                           ::cuda::std::tuple<SourceIteratorTs...>,
                                                                           BindStagedSourcesOpT,
                                                                           SourceOffsetOpT>>
{
  using iterator_t =
    transform_prolog_stageable_iterator<LogicalIteratorT,
                                        ::cuda::std::tuple<SourceIteratorTs...>,
                                        BindStagedSourcesOpT,
                                        SourceOffsetOpT>;
  using staged_source_tuple_t = ::cuda::std::tuple<transform_prolog_staged_iterator_t<SourceIteratorTs>...>;
  using staged_iterator_t     = ::cuda::std::decay_t<decltype(::cuda::std::declval<BindStagedSourcesOpT>()(
    ::cuda::std::declval<LogicalIteratorT>(),
    ::cuda::std::declval<staged_source_tuple_t>(),
    ::cuda::std::declval<::cuda::std::size_t>()))>;
  using source_state_tuple_t =
    ::cuda::std::tuple<typename transform_prolog_staging_traits_t<SourceIteratorTs>::staging_state_t...>;

  struct staging_state_t
  {
    source_state_tuple_t source_states{};
    ::cuda::std::size_t absolute_logical_offset{0};
  };

  static constexpr bool is_supported = (... && transform_prolog_stageable_input_v<SourceIteratorTs>);
  static constexpr int source_count  = (transform_prolog_staging_traits_t<SourceIteratorTs>::source_count + ... + 0);

  _CCCL_HOST_DEVICE static constexpr int shared_buffer_alignment(int bulk_copy_alignment)
  {
    return transform_prolog_staging_tuple_alignment<SourceIteratorTs...>(bulk_copy_alignment);
  }

  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t shared_buffer_size(int items, int bulk_copy_alignment)
  {
    return transform_prolog_staging_tuple_size<SourceIteratorTs...>(items, bulk_copy_alignment);
  }

  template <typename IssueCopyAsyncT, ::cuda::std::size_t... Is>
  _CCCL_DEVICE static void issue_load_to_shared_impl(
    iterator_t it,
    ::cuda::std::size_t absolute_logical_offset,
    int items,
    char* shared_buffer_base,
    ::cuda::std::size_t& shared_buffer_offset,
    staging_state_t& staging_state,
    IssueCopyAsyncT&& issue_copy_async,
    ::cuda::std::index_sequence<Is...>)
  {
    (
      [&] {
        using source_iterator_t = ::cuda::std::tuple_element_t<Is, ::cuda::std::tuple<SourceIteratorTs...>>;
        using source_traits_t   = transform_prolog_staging_traits_t<source_iterator_t>;

        source_traits_t::issue_load_to_shared(
          ::cuda::std::get<Is>(it.source_iterators()),
          it.source_offset(::cuda::std::integral_constant<::cuda::std::size_t, Is>{}, absolute_logical_offset, items),
          items,
          shared_buffer_base,
          shared_buffer_offset,
          ::cuda::std::get<Is>(staging_state.source_states),
          issue_copy_async);
      }(),
      ...);
  }

  template <typename IssueCopyAsyncT>
  _CCCL_DEVICE static void issue_load_to_shared(
    iterator_t it,
    ::cuda::std::size_t logical_offset,
    int items,
    char* shared_buffer_base,
    ::cuda::std::size_t& shared_buffer_offset,
    staging_state_t& staging_state,
    IssueCopyAsyncT&& issue_copy_async)
  {
    const auto absolute_logical_offset    = it.logical_base_offset() + logical_offset;
    staging_state.absolute_logical_offset = absolute_logical_offset;

    issue_load_to_shared_impl(
      it,
      absolute_logical_offset,
      items,
      shared_buffer_base,
      shared_buffer_offset,
      staging_state,
      ::cuda::std::forward<IssueCopyAsyncT>(issue_copy_async),
      ::cuda::std::index_sequence_for<SourceIteratorTs...>{});
  }

  template <::cuda::std::size_t... Is>
  _CCCL_DEVICE static staged_source_tuple_t
  bind_staged_sources_impl(iterator_t it, const staging_state_t& staging_state, ::cuda::std::index_sequence<Is...>)
  {
    return ::cuda::std::make_tuple(([&] {
      using source_iterator_t = ::cuda::std::tuple_element_t<Is, ::cuda::std::tuple<SourceIteratorTs...>>;
      using source_traits_t   = transform_prolog_staging_traits_t<source_iterator_t>;

      return source_traits_t::bind_shared_input(
        ::cuda::std::get<Is>(it.source_iterators()), ::cuda::std::get<Is>(staging_state.source_states));
    }())...);
  }

  _CCCL_DEVICE static staged_iterator_t bind_shared_input(iterator_t it, const staging_state_t& staging_state)
  {
    return it.bind_staged_sources(
      bind_staged_sources_impl(it, staging_state, ::cuda::std::index_sequence_for<SourceIteratorTs...>{}),
      staging_state.absolute_logical_offset);
  }
};

template <typename IteratorTupleT>
struct transform_prolog_staging_traits<THRUST_NS_QUALIFIER::zip_iterator<IteratorTupleT>>
    : transform_prolog_zip_staging_traits<IteratorTupleT>
{
  using iterator_t = THRUST_NS_QUALIFIER::zip_iterator<IteratorTupleT>;
  using base_t     = transform_prolog_zip_staging_traits<IteratorTupleT>;

  template <typename IssueCopyAsyncT>
  _CCCL_DEVICE static void issue_load_to_shared(
    iterator_t it,
    ::cuda::std::size_t logical_offset,
    int items,
    char* shared_buffer_base,
    ::cuda::std::size_t& shared_buffer_offset,
    typename base_t::staging_state_t& staging_state,
    IssueCopyAsyncT&& issue_copy_async)
  {
    base_t::issue_load_to_shared_impl(
      it.get_iterator_tuple(),
      logical_offset,
      items,
      shared_buffer_base,
      shared_buffer_offset,
      staging_state,
      ::cuda::std::forward<IssueCopyAsyncT>(issue_copy_async),
      ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<IteratorTupleT>>{});
  }

  _CCCL_DEVICE static typename base_t::staged_iterator_t
  bind_shared_input(iterator_t it, const typename base_t::staging_state_t& staging_state)
  {
    return base_t::bind_shared_input_impl(
      it.get_iterator_tuple(),
      staging_state,
      ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<IteratorTupleT>>{});
  }
};

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int segment_size,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Initial block-only implementation:
  // - one block owns one fixed-size contiguous segment
  // - each thread receives a contiguous slice of that segment via `thread_segment_range`
  // - `segment_op` is responsible for any block-wide combine it needs and should return the final segment result on
  //   every thread in the block group
  // - the block first stages the segment into shared memory via tiled `BlockLoad`
  // - the kernel then applies `element_transform_op` to each item, passing the segment result, the segment-local item
  //   index, and the item value

  using block_hierarchy_t  = decltype(::cuda::hierarchy(::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using block_group_t      = ::cuda::experimental::this_block<block_hierarchy_t>;
  using input_value_t      = cub::detail::it_value_t<InputIteratorT>;
  using output_value_t     = cub::detail::non_void_value_t<OutputIteratorT, input_value_t>;
  using staging_traits_t   = transform_prolog_staging_traits_t<InputIteratorT>;
  using staged_iterator_t  = typename staging_traits_t::staged_iterator_t;
  using segment_iterator_t = transform_prolog_segment_iterator_t<staged_iterator_t>;
  using input_range_t      = thread_segment_range<segment_iterator_t>;
  using segment_result_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, block_group_t, input_range_t>>;
  using block_load_to_shared_t = BlockLoadToShared<BlockThreads>;
  using block_store_t          = BlockStore<output_value_t, BlockThreads, ItemsPerThread, BLOCK_STORE_STRIPED>;

  // There is a subtle difference with the epilog case. There we did
  // IPT because each thread was doing ipt items. Here each thread
  // does do IPT items but it also loads other stuff so it cal
  // calculate the RMS. So here, BlockLoad does not need to be tied to
  // IPT in the same way.
  constexpr int tile_items = BlockThreads * ItemsPerThread;

  static_assert(transform_prolog_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input iterators to expose stageable contiguous source(s).");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");
  extern __shared__ char shared_segment_buffer_base[];
  __shared__ typename block_load_to_shared_t::TempStorage load_to_shared_storage;

  const int segment_id = static_cast<int>(blockIdx.x);
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const int thread_rank = static_cast<int>(threadIdx.x);

  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group             = ::cuda::experimental::this_block{block_hierarchy};
  auto apply_element_transform = [&](const segment_result_t& segment_result, int index_in_segment, auto&& value) {
    using input_ref_t = decltype(value);

    if constexpr (::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, int, input_ref_t>)
    {
      return element_transform_op(segment_result, index_in_segment, static_cast<input_ref_t>(value));
    }
    else
    {
      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, input_ref_t>,
                    "element_transform_op must be invocable with either "
                    "(segment_result, index_in_segment, value) or (segment_result, value).");
      return element_transform_op(segment_result, static_cast<input_ref_t>(value));
    }
  };

  constexpr int shared_buffer_alignment =
    staging_traits_t::shared_buffer_alignment(transform_prolog_bulk_copy_alignment);
  char* aligned_shared_buffer = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
  auto staged_segment         = [&] {
    block_load_to_shared_t load_to_shared{load_to_shared_storage};
    ::cuda::std::size_t shared_buffer_offset = 0;
    typename staging_traits_t::staging_state_t staging_state{};
    auto issue_copy_async = [&](auto shared_buffer, auto input_buffer) {
      load_to_shared.CopyAsync(shared_buffer, input_buffer);
    };

    staging_traits_t::issue_load_to_shared(
      d_in, segment_offset, segment_size, aligned_shared_buffer, shared_buffer_offset, staging_state, issue_copy_async);

    if constexpr (staging_traits_t::source_count > 0)
    {
      if (segment_size > 0)
      {
        auto token  = load_to_shared.Commit();
        auto staged = staging_traits_t::bind_shared_input(d_in, staging_state);
        load_to_shared.Wait(::cuda::std::move(token));
        return staged;
      }
    }

    return staging_traits_t::bind_shared_input(d_in, staging_state);
  }();

  block_group.sync();

  auto segment_input                    = transform_prolog_segment_input(staged_segment);
  auto input_range                      = make_thread_segment_range<BlockThreads>(segment_input, segment_size);
  const segment_result_t segment_result = segment_op(block_group, input_range);

  for (int tile_base = 0; tile_base < segment_size; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, segment_size - tile_base);
    output_value_t output_items[ItemsPerThread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;
      output_items[item]         = output_value_t{};

      if (tile_local_index < valid_items)
      {
        const int index_in_segment = tile_base + tile_local_index;
        output_items[item] =
          apply_element_transform(segment_result, index_in_segment, staged_segment[index_in_segment]);
      }
    }

    block_store_t().Store(d_out + segment_offset + tile_base, output_items, valid_items);
  }
}

template <int BlockThreads,
          int ItemsPerThread,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename SegmentOpT,
          typename ElementTransformOpT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(BlockThreads) void DeviceHierarchicalTransformClusterKernel(
  _CCCL_GRID_CONSTANT const InputIteratorT d_in,
  _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
  _CCCL_GRID_CONSTANT const int segment_size,
  _CCCL_GRID_CONSTANT const int cluster_size,
  _CCCL_GRID_CONSTANT const int chunk_items,
  SegmentOpT segment_op,
  ElementTransformOpT element_transform_op)
{
  // Large-segment path:
  // - one cluster owns one fixed-size contiguous segment
  // - each CTA in the cluster stages one contiguous segment chunk in its local shared memory
  // - the segment op sees a cluster group and the CTA-local chunk range
  // - the segment op is responsible for reducing across the cluster, for example through cuda::device::reduce
  // - each CTA transforms and stores only the items from its own staged chunk

  using block_hierarchy_t = decltype(::cuda::hierarchy(::cuda::grid_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using cluster_hierarchy_t = decltype(::cuda::hierarchy(
    ::cuda::grid_dims(dim3{}), ::cuda::cluster_dims(dim3{}), ::cuda::block_dims<BlockThreads>()));
  using cluster_group_t     = ::cuda::experimental::this_cluster<cluster_hierarchy_t>;
  using input_value_t       = cub::detail::it_value_t<InputIteratorT>;
  using output_value_t      = cub::detail::non_void_value_t<OutputIteratorT, input_value_t>;
  using staging_traits_t    = transform_prolog_staging_traits_t<InputIteratorT>;
  using staged_iterator_t   = typename staging_traits_t::staged_iterator_t;
  using segment_iterator_t  = transform_prolog_segment_iterator_t<staged_iterator_t>;
  using input_range_t       = thread_segment_range<segment_iterator_t>;
  using segment_result_t =
    ::cuda::std::decay_t<::cuda::std::invoke_result_t<SegmentOpT, cluster_group_t, input_range_t>>;
  using block_load_to_shared_t = BlockLoadToShared<BlockThreads>;
  using block_store_t          = BlockStore<output_value_t, BlockThreads, ItemsPerThread, BLOCK_STORE_STRIPED>;

  constexpr int tile_items = BlockThreads * ItemsPerThread;

  static_assert(transform_prolog_stageable_input_v<InputIteratorT>,
                "TransformProlog requires input iterators to expose stageable contiguous source(s).");
  static_assert(!::cuda::std::is_void_v<segment_result_t>, "segment_op must return one scalar result per segment.");

  extern __shared__ char shared_segment_buffer_base[];
  __shared__ typename block_load_to_shared_t::TempStorage load_to_shared_storage;

  const int cluster_rank = static_cast<int>(blockIdx.x % cluster_size);
  const int segment_id   = static_cast<int>(blockIdx.x / cluster_size);
  const int chunk_begin  = (::cuda::std::min) (cluster_rank * chunk_items, segment_size);
  const int chunk_end    = (::cuda::std::min) (chunk_begin + chunk_items, segment_size);
  const int local_items  = chunk_end - chunk_begin;
  const auto segment_offset =
    static_cast<::cuda::std::size_t>(segment_id) * static_cast<::cuda::std::size_t>(segment_size);
  const int thread_rank = static_cast<int>(threadIdx.x);

  const auto block_hierarchy   = ::cuda::hierarchy(::cuda::grid_dims(gridDim), ::cuda::block_dims<BlockThreads>());
  auto block_group             = ::cuda::experimental::this_block{block_hierarchy};
  const auto cluster_hierarchy = ::cuda::hierarchy(
    ::cuda::grid_dims(gridDim),
    ::cuda::cluster_dims(dim3(static_cast<unsigned int>(cluster_size), 1, 1)),
    ::cuda::block_dims<BlockThreads>());
  auto cluster_group = ::cuda::experimental::this_cluster{cluster_hierarchy};

  auto apply_element_transform = [&](const segment_result_t& segment_result, int index_in_segment, auto&& value) {
    using input_ref_t = decltype(value);

    if constexpr (::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, int, input_ref_t>)
    {
      return element_transform_op(segment_result, index_in_segment, static_cast<input_ref_t>(value));
    }
    else
    {
      static_assert(::cuda::std::is_invocable_v<ElementTransformOpT, segment_result_t, input_ref_t>,
                    "element_transform_op must be invocable with either "
                    "(segment_result, index_in_segment, value) or (segment_result, value).");
      return element_transform_op(segment_result, static_cast<input_ref_t>(value));
    }
  };

  constexpr int shared_buffer_alignment =
    staging_traits_t::shared_buffer_alignment(transform_prolog_bulk_copy_alignment);
  char* aligned_shared_buffer = align_dynamic_shared_buffer<shared_buffer_alignment>(shared_segment_buffer_base);
  auto staged_chunk           = [&] {
    block_load_to_shared_t load_to_shared{load_to_shared_storage};
    ::cuda::std::size_t shared_buffer_offset = 0;
    typename staging_traits_t::staging_state_t staging_state{};
    auto issue_copy_async = [&](auto shared_buffer, auto input_buffer) {
      load_to_shared.CopyAsync(shared_buffer, input_buffer);
    };

    staging_traits_t::issue_load_to_shared(
      d_in,
      segment_offset + static_cast<::cuda::std::size_t>(chunk_begin),
      local_items,
      aligned_shared_buffer,
      shared_buffer_offset,
      staging_state,
      issue_copy_async);

    if constexpr (staging_traits_t::source_count > 0)
    {
      if (local_items > 0)
      {
        auto token  = load_to_shared.Commit();
        auto staged = staging_traits_t::bind_shared_input(d_in, staging_state);
        load_to_shared.Wait(::cuda::std::move(token));
        return staged;
      }
    }

    return staging_traits_t::bind_shared_input(d_in, staging_state);
  }();

  block_group.sync();

  auto segment_input     = transform_prolog_segment_input(staged_chunk);
  const auto local_range = make_thread_segment_range<BlockThreads>(segment_input, local_items);
  auto input_range       = input_range_t{local_range.begin(), chunk_begin + local_range.offset(), local_range.size()};
  const segment_result_t segment_result = segment_op(cluster_group, input_range);

  for (int tile_base = 0; tile_base < local_items; tile_base += tile_items)
  {
    const int valid_items = (::cuda::std::min) (tile_items, local_items - tile_base);
    output_value_t output_items[ItemsPerThread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int tile_local_index = thread_rank + item * BlockThreads;
      output_items[item]         = output_value_t{};

      if (tile_local_index < valid_items)
      {
        const int local_index      = tile_base + tile_local_index;
        const int index_in_segment = chunk_begin + local_index;
        output_items[item] = apply_element_transform(segment_result, index_in_segment, staged_chunk[local_index]);
      }
    }

    block_store_t().Store(d_out + segment_offset + chunk_begin + tile_base, output_items, valid_items);
  }
}
} // namespace detail::hierarchical

CUB_NAMESPACE_END
