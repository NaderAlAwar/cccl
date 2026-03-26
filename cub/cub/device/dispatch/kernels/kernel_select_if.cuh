// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_select_if.cuh>
#include <cub/util_vsmem.cuh>

#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstdint>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored

CUB_NAMESPACE_BEGIN

namespace detail::select
{
// Offset type used to instantiate the stream compaction-kernel and agent to index the items within one partition.
using per_partition_offset_t = ::cuda::std::int32_t;

template <typename TotalNumItemsT, bool IsStreamingInvocation>
class streaming_context_t
{
private:
  bool first_partition = true;
  bool last_partition  = false;
  TotalNumItemsT total_num_items{};
  TotalNumItemsT total_previous_num_items{};

  // Double-buffer the number of previously selected items across partitions.
  TotalNumItemsT* d_num_selected_in  = nullptr;
  TotalNumItemsT* d_num_selected_out = nullptr;

public:
  using total_num_items_t = TotalNumItemsT;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE streaming_context_t(
    TotalNumItemsT* d_num_selected_in,
    TotalNumItemsT* d_num_selected_out,
    TotalNumItemsT total_num_items,
    bool is_last_partition)
      : last_partition(is_last_partition)
      , total_num_items(total_num_items)
      , d_num_selected_in(d_num_selected_in)
      , d_num_selected_out(d_num_selected_out)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(TotalNumItemsT num_items, bool next_partition_is_the_last)
  {
    using ::cuda::std::swap;
    swap(d_num_selected_in, d_num_selected_out);
    first_partition = false;
    last_partition  = next_partition_is_the_last;
    total_previous_num_items += num_items;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT input_offset() const
  {
    return first_partition ? TotalNumItemsT{0} : total_previous_num_items;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT is_first_partition() const
  {
    return first_partition;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_selected() const
  {
    return first_partition ? TotalNumItemsT{0} : *d_num_selected_in;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_rejected() const
  {
    return first_partition ? TotalNumItemsT{0} : (total_previous_num_items - num_previously_selected());
  }

  template <typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_total_items(OffsetT) const
  {
    return total_num_items;
  }

  template <typename NumSelectedIteratorT, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
  update_num_selected(NumSelectedIteratorT user_num_selected_out_it, OffsetT num_selections) const
  {
    if (last_partition)
    {
      *user_num_selected_out_it = num_previously_selected() + static_cast<TotalNumItemsT>(num_selections);
    }
    else
    {
      *d_num_selected_out = num_previously_selected() + static_cast<TotalNumItemsT>(num_selections);
    }
  }
};

template <typename TotalNumItemsT>
class streaming_context_t<TotalNumItemsT, false>
{
public:
  using total_num_items_t = TotalNumItemsT;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE streaming_context_t(TotalNumItemsT*, TotalNumItemsT*, TotalNumItemsT, bool) {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(TotalNumItemsT, bool) {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT input_offset() const
  {
    return TotalNumItemsT{0};
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT is_first_partition() const
  {
    return true;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_selected() const
  {
    return TotalNumItemsT{0};
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_rejected() const
  {
    return TotalNumItemsT{0};
  }

  template <typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_total_items(OffsetT num_partition_items) const
  {
    return num_partition_items;
  }

  template <typename NumSelectedIteratorT, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
  update_num_selected(NumSelectedIteratorT user_num_selected_out_it, OffsetT num_selections) const
  {
    *user_num_selected_out_it = num_selections;
  }
};

template <SelectImpl SelectionOpt>
struct agent_select_if_wrapper_t
{
  template <typename AgentSelectIfPolicyT,
            typename InputIteratorT,
            typename FlagsInputIteratorT,
            typename SelectedOutputIteratorT,
            typename SelectOpT,
            typename EqualityOpT,
            typename OffsetT,
            typename StreamingContextT>
  struct agent_t
      : AgentSelectIf<AgentSelectIfPolicyT,
                      InputIteratorT,
                      FlagsInputIteratorT,
                      SelectedOutputIteratorT,
                      SelectOpT,
                      EqualityOpT,
                      OffsetT,
                      StreamingContextT,
                      SelectionOpt>
  {
    using AgentSelectIf<AgentSelectIfPolicyT,
                        InputIteratorT,
                        FlagsInputIteratorT,
                        SelectedOutputIteratorT,
                        SelectOpT,
                        EqualityOpT,
                        OffsetT,
                        StreamingContextT,
                        SelectionOpt>::AgentSelectIf;
  };
};

template <SelectImpl SelectionOpt>
struct VSMemHelper
{
  template <typename ActivePolicyT, typename... Ts>
  using VSMemHelperDefaultFallbackPolicyT =
    vsmem_helper_default_fallback_policy_t<ActivePolicyT, agent_select_if_wrapper_t<SelectionOpt>::template agent_t, Ts...>;

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr int BlockThreads(ActivePolicyT)
  {
    return VSMemHelperDefaultFallbackPolicyT<ActivePolicyT, Ts...>::agent_policy_t::BLOCK_THREADS;
  }

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr int ItemsPerThread(ActivePolicyT)
  {
    return VSMemHelperDefaultFallbackPolicyT<ActivePolicyT, Ts...>::agent_policy_t::ITEMS_PER_THREAD;
  }

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t VSMemPerBlock(ActivePolicyT)
  {
    return VSMemHelperDefaultFallbackPolicyT<ActivePolicyT, Ts...>::vsmem_per_block;
  }
};

/**
 * @brief Wrapper around the select/partition sweep kernel.
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename FlagsInputIteratorT,
          typename SelectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectOpT,
          typename EqualityOpT,
          typename OffsetT,
          typename StreamingContextT,
          SelectImpl SelectionOpt>
__launch_bounds__(int(
  VSMemHelper<SelectionOpt>::template VSMemHelperDefaultFallbackPolicyT<
    typename ChainedPolicyT::ActivePolicy::SelectIfPolicyT,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    SelectOpT,
    EqualityOpT,
    OffsetT,
    StreamingContextT>::agent_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSelectSweepKernel(
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_status,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles,
    _CCCL_GRID_CONSTANT const StreamingContextT streaming_context,
    vsmem_t vsmem)
{
  using VsmemHelperT = typename VSMemHelper<SelectionOpt>::template VSMemHelperDefaultFallbackPolicyT<
    typename ChainedPolicyT::ActivePolicy::SelectIfPolicyT,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    SelectOpT,
    EqualityOpT,
    OffsetT,
    StreamingContextT>;

  using AgentSelectIfPolicyT = typename VsmemHelperT::agent_policy_t;
  using AgentSelectIfT       = typename VsmemHelperT::agent_t;

  __shared__ typename VsmemHelperT::static_temp_storage_t static_temp_storage;

  typename AgentSelectIfT::TempStorage& temp_storage = VsmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  AgentSelectIfT(temp_storage, d_in, d_flags, d_selected_out, select_op, equality_op, num_items, streaming_context)
    .ConsumeRange(num_tiles, tile_status, d_num_selected_out);

  VsmemHelperT::discard_temp_storage(temp_storage);
}
} // namespace detail::select

CUB_NAMESPACE_END

_CCCL_DIAG_POP
