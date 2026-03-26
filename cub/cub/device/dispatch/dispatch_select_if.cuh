// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 *   cub::DeviceSelect provides device-wide, parallel operations for selecting items from sequences
 *   of data items residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_select_if.cuh>
#include <cub/device/dispatch/tuning/tuning_select_if.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/limits>

#include <nv/target>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

CUB_NAMESPACE_BEGIN

namespace detail::select
{
template <typename MaxPolicyT,
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
struct DeviceSelectIfKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(CompactInitKernel, detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>);

  CUB_DEFINE_KERNEL_GETTER(
    SelectIfKernel,
    DeviceSelectSweepKernel<
      MaxPolicyT,
      InputIteratorT,
      FlagsInputIteratorT,
      SelectedOutputIteratorT,
      NumSelectedIteratorT,
      ScanTileStateT,
      SelectOpT,
      EqualityOpT,
      OffsetT,
      StreamingContextT,
      SelectionOpt>);
};

template <SelectImpl SelectionOpt, typename OffsetT>
inline constexpr bool use_streaming_context_v =
  (SelectionOpt != SelectImpl::Partition)
  || (static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<per_partition_offset_t>::max())
      < static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<OffsetT>::max()));
} // namespace detail::select

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSelect and DevicePartition
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items
 *
 * @tparam FlagsInputIteratorT
 *   Random-access input iterator type for reading selection flags (NullType* if a selection functor or discontinuity
 *   flagging is used for selection)
 *
 * @tparam SelectedOutputIteratorT
 *   Random-access output iterator type for writing selected items
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam SelectOpT
 *   Selection operator type (NullType if selection flags or discontinuity flagging is used for selection)
 *
 * @tparam EqualityOpT
 *   Equality operator type (NullType if selection functor or selection flags are used for selection)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam SelectionOpt
 *   SelectImpl indicating whether to partition, just selection or selection where the memory for the input and
 *   output may alias each other.
 */
template <
  typename InputIteratorT,
  typename FlagsInputIteratorT,
  typename SelectedOutputIteratorT,
  typename NumSelectedIteratorT,
  typename SelectOpT,
  typename EqualityOpT,
  typename OffsetT,
  SelectImpl SelectionOpt,
  typename PolicyHub = detail::select::policy_hub<
    detail::it_value_t<InputIteratorT>,
    detail::it_value_t<FlagsInputIteratorT>,
    // if/flagged/unique only have a single code path for different offset types, partition has different code paths
    ::cuda::std::conditional_t<SelectionOpt == SelectImpl::Partition, OffsetT, detail::select::per_partition_offset_t>,
    detail::select::is_partition_distinct_output_t<SelectedOutputIteratorT>::value,
    SelectionOpt>,
  typename KernelSource = detail::select::DeviceSelectIfKernelSource<
    typename PolicyHub::MaxPolicy,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    NumSelectedIteratorT,
    ScanTileState<detail::select::per_partition_offset_t>,
    SelectOpT,
    EqualityOpT,
    detail::select::per_partition_offset_t,
    detail::select::streaming_context_t<OffsetT, detail::select::use_streaming_context_v<SelectionOpt, OffsetT>>,
    SelectionOpt>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
  typename VSMemHelperT          = detail::select::VSMemHelper<SelectionOpt>>
struct DispatchSelectIf
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/

  // Offset type used to instantiate the stream compaction-kernel and agent to index the items within one partition
  using per_partition_offset_t = detail::select::per_partition_offset_t;

  // Offset type large enough to represent any index within the input and output iterators
  using num_total_items_t = OffsetT;

  // Whether the algorithm is a partitioning invocation (versus a selection invocation)
  static constexpr bool is_partitioning_invocation = (SelectionOpt == SelectImpl::Partition);

  // We always use a streaming context for selection. However, for a partitioning invocation, we only use a streaming
  // context when necessary. I.e., if the values representable by OffsetT exceed the values representable by
  // per_partition_offset_t.
  static constexpr bool use_streaming_context = detail::select::use_streaming_context_v<SelectionOpt, OffsetT>;

  using streaming_context_t = detail::select::streaming_context_t<num_total_items_t, use_streaming_context>;

  using ScanTileStateT = ScanTileState<per_partition_offset_t>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes`
  /// and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the input sequence of selection flags (if applicable)
  FlagsInputIteratorT d_flags;

  /// Pointer to the output sequence of selected data items
  SelectedOutputIteratorT d_selected_out;

  /// Pointer to the total number of items selected (i.e., length of `d_selected_out`)
  NumSelectedIteratorT d_num_selected_out;

  /// Selection operator
  SelectOpT select_op;

  /// Equality operator
  EqualityOpT equality_op;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  /**
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When `nullptr`, the required allocation size is written to `temp_storage_bytes`
   *   and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_flags
   *   Pointer to the input sequence of selection flags (if applicable)
   *
   * @param d_selected_out
   *   Pointer to the output sequence of selected data items
   *
   * @param d_num_selected_out
   *  Pointer to the total number of items selected (i.e., length of `d_selected_out`)
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSelectIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_flags(d_flags)
      , d_selected_out(d_selected_out)
      , d_num_selected_out(d_num_selected_out)
      , select_op(select_op)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  /**
   * Internal dispatch routine for computing a device-wide selection using the
   * specified kernel functions.
   */
  template <typename ActivePolicyT, typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(ActivePolicyT policy, ScanInitKernelPtrT scan_init_kernel, SelectIfKernelPtrT select_if_kernel)
  {
    const auto block_threads = VSMemHelperT::template BlockThreads<
      typename ActivePolicyT::SelectIfPolicyT,
      InputIteratorT,
      FlagsInputIteratorT,
      SelectedOutputIteratorT,
      SelectOpT,
      EqualityOpT,
      per_partition_offset_t,
      streaming_context_t>(policy.SelectIf());
    const auto items_per_thread = VSMemHelperT::template ItemsPerThread<
      typename ActivePolicyT::SelectIfPolicyT,
      InputIteratorT,
      FlagsInputIteratorT,
      SelectedOutputIteratorT,
      SelectOpT,
      EqualityOpT,
      per_partition_offset_t,
      streaming_context_t>(policy.SelectIf());
    const auto tile_size = static_cast<OffsetT>(block_threads * items_per_thread);

    // The maximum number of items per partition
    constexpr auto max_supported_partition_size = ::cuda::std::numeric_limits<per_partition_offset_t>::max();
    const auto full_tile_partition_size = static_cast<per_partition_offset_t>(
      max_supported_partition_size - (max_supported_partition_size % static_cast<per_partition_offset_t>(tile_size)));

    // For partitioning invocations, we cap the partition size to the maximum number of items supported.
    // For selection invocations, we cap at the largest multiple of a full tile. There's a selection-specific bug where
    // we would otherwise overflow indices for the last partial tile, when discounting for the out-of-bounds items.
    const per_partition_offset_t capped_partition_size =
      is_partitioning_invocation ? max_supported_partition_size : full_tile_partition_size;

    // The maximum number of items for which we will ever invoke the kernel (i.e. largest partition size)
    // The extra check of use_streaming_context ensures that OffsetT is larger than per_partition_offset_t to avoid
    // truncation of partition_size
    auto const max_partition_size =
      (use_streaming_context && num_items > static_cast<OffsetT>(capped_partition_size))
        ? static_cast<OffsetT>(capped_partition_size)
        : num_items;

    // The number of partitions required to "iterate" over the total input (ternary to avoid div-by-zero)
    auto const num_partitions =
      (max_partition_size == 0) ? static_cast<OffsetT>(1) : ::cuda::ceil_div(num_items, max_partition_size);

    // The maximum number of tiles for which we will ever invoke the kernel
    auto const max_num_tiles_per_invocation = static_cast<OffsetT>(::cuda::ceil_div(max_partition_size, tile_size));

    // The amount of virtual shared memory to allocate
    const auto vsmem_size =
      max_num_tiles_per_invocation
      * VSMemHelperT::template VSMemPerBlock<
        typename ActivePolicyT::SelectIfPolicyT,
        InputIteratorT,
        FlagsInputIteratorT,
        SelectedOutputIteratorT,
        SelectOpT,
        EqualityOpT,
        per_partition_offset_t,
        streaming_context_t>(policy.SelectIf());

    cudaError error = cudaSuccess;

    // Specify temporary storage allocation requirements
    ::cuda::std::size_t streaming_selection_storage_bytes =
      (num_partitions > 1) ? 2 * sizeof(num_total_items_t) : ::cuda::std::size_t{0};
    ::cuda::std::size_t allocation_sizes[3] = {0ULL, vsmem_size, streaming_selection_storage_bytes};

    // Bytes needed for tile status descriptors
    error = CubDebug(ScanTileStateT::AllocationSize(static_cast<int>(max_num_tiles_per_invocation), allocation_sizes[0]));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[3] = {};

    error = CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation.
      return cudaSuccess;
    }

    // Initialize the streaming context with the temporary storage for double-buffering the previously selected items
    // and the total number (across all partitions) of items.
    num_total_items_t* tmp_num_selected_out = reinterpret_cast<num_total_items_t*>(allocations[2]);
    streaming_context_t streaming_context{
      tmp_num_selected_out, (tmp_num_selected_out + 1), num_items, (num_partitions <= 1)};

    // Iterate over the partitions until all input is processed.
    for (OffsetT partition_idx = 0; partition_idx < num_partitions; partition_idx++)
    {
      OffsetT current_partition_offset = partition_idx * max_partition_size;
      OffsetT current_num_items =
        (partition_idx + 1 == num_partitions) ? (num_items - current_partition_offset) : max_partition_size;

      const auto current_num_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));
      ScanTileStateT tile_status;
      error = CubDebug(tile_status.Init(current_num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        return error;
      }

      const int init_grid_size = ::cuda::std::max(1, ::cuda::ceil_div(current_num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking scan_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              reinterpret_cast<long long>(stream));
#endif

      launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(scan_init_kernel, tile_status, current_num_tiles, d_num_selected_out);

      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }

      // No more items to process (note, we do not want to return early for num_items == 0, because we need to make
      // sure that `scan_init_kernel` has written '0' to d_num_selected_out).
      if (current_num_items == 0)
      {
        return cudaSuccess;
      }

#ifdef CUB_DEBUG_LOG
      {
        int range_select_sm_occupancy;
        error = CubDebug(launcher_factory.MaxSmOccupancy(range_select_sm_occupancy, select_if_kernel, block_threads));
        if (cudaSuccess != error)
        {
          return error;
        }

        _CubLog("Invoking select_if_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                current_num_tiles,
                block_threads,
                reinterpret_cast<long long>(stream),
                items_per_thread,
                range_select_sm_occupancy);
      }
#endif

      launcher_factory(current_num_tiles, block_threads, 0, stream)
        .doit(select_if_kernel,
              d_in,
              d_flags,
              d_selected_out,
              d_num_selected_out,
              tile_status,
              select_op,
              equality_op,
              static_cast<per_partition_offset_t>(current_num_items),
              current_num_tiles,
              streaming_context,
              cub::detail::vsmem_t{allocations[1]});

      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }

      streaming_context.advance(current_num_items, (partition_idx + OffsetT{2} == num_partitions));
    }

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    const auto wrapped_policy = detail::select::MakeSelectIfPolicyWrapper(active_policy);
    return Invoke(wrapped_policy, kernel_source.CompactInitKernel(), kernel_source.SelectIfKernel());
  }

  /**
   * Internal dispatch routine
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When `nullptr`, the required allocation size is written to `temp_storage_bytes`
   *   and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_flags
   *   Pointer to the input sequence of selection flags (if applicable)
   *
   * @param d_selected_out
   *   Pointer to the output sequence of selected data items
   *
   * @param d_num_selected_out
   *  Pointer to the total number of items selected (i.e., length of `d_selected_out`)
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    int ptx_version = 0;
    if (cudaError_t error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchSelectIf dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_flags,
      d_selected_out,
      d_num_selected_out,
      select_op,
      equality_op,
      num_items,
      stream,
      kernel_source,
      launcher_factory);

    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
  }
};

CUB_NAMESPACE_END

_CCCL_DIAG_POP
