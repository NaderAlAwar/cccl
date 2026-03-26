# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numba.cuda
import numpy as np
import nvtx
from cuda.compute import (
    CountingIterator,
    OpKind,
    PermutationIterator,
    TransformIterator,
    ZipIterator,
    exclusive_scan,
    select,
    unary_transform,
)


@nvtx.annotate("segment_sizes")
def segment_sizes(offsets):
    """
    Compute the size of each segment from segment offsets.

    Args:
        offsets: Device array of segment offsets (length = num_segments + 1).
                 Each segment i contains elements from offsets[i] to offsets[i+1].

    Returns:
        Device array of segment sizes (length = num_segments).
    """
    return offsets[1:] - offsets[:-1]


@nvtx.annotate("offsets_to_segment_ids")
def offsets_to_segment_ids(offsets, stream=None):
    """
    Convert segment offsets to segment IDs (indicators).

    Given offsets [0, 2, 5, 8, 10], produces [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

    This function correctly handles empty segments (duplicate offsets).
    For example, offsets [0, 0, 2, 3] with an empty first segment produces [0, 0, 1]
    where elements at indices 0-1 belong to segment 1 (the empty segment 0 has no elements).

    The implementation uses binary search to find which segment each element belongs to.
    For each element at index i, we find segment j where offsets[j] <= i < offsets[j+1].

    Args:
        offsets: Device array of segment offsets (length = num_segments + 1).
                 The last element is the total number of elements.
        stream: CUDA stream for the operation (optional).

    Returns:
        Device array of segment IDs for each element (length = offsets[-1]).
    """
    num_elements = int(offsets[-1])

    if num_elements == 0:
        return cp.array([], dtype=np.int32)

    # Create array of all element indices [0, 1, 2, ..., num_elements-1]
    element_indices = cp.arange(num_elements, dtype=np.int32)

    # Use binary search to find which segment each element belongs to
    # searchsorted finds the rightmost position where we can insert each element index
    # such that the array remains sorted. Subtracting 1 gives us the segment ID.
    #
    # Example: offsets = [0, 0, 2, 3], element_indices = [0, 1, 2]
    # searchsorted(offsets[:-1], [0, 1, 2], side='right') = [1, 1, 2]
    # Subtracting 1: [0, 0, 1] - but wait, element 0 should be in segment 1!
    #
    # Actually, we want: offsets[j] <= i < offsets[j+1]
    # So searchsorted with side='right' on offsets (not offsets[:-1]) and subtract 1
    segment_ids = cp.searchsorted(offsets, element_indices, side="right") - 1

    return segment_ids


@nvtx.annotate("select_segments")
def select_segments(
    data_in,
    offsets_in,
    mask_in,
    data_out,
    offsets_out,
    d_num_selected_out,
    num_elements,
    num_segments,
    stream=None,
):
    """
    Select segments of a segmented array using a per-segment mask.

    A segmented array is conceptually composed of a data array and segment offsets.
    For example, with data=[30, 20, 20, 50, 90, 10, 30, 80, 20, 60] and
    offsets=[0, 2, 5, 8, 10], the segmented array represents:
    [[30, 20], [20, 50, 90], [10, 30, 80], [20, 60]]

    This function selects entire segments based on a mask, keeping only
    the segments where mask[i] is non-zero. The results are written to the provided
    output arrays/iterators.

    Example:
        >>> data_in = cp.array([30, 20, 20, 50, 90, 10, 30, 80, 20, 60], dtype=np.int32)
        >>> offsets_in = cp.array([0, 2, 5, 8, 10], dtype=np.int32)
        >>> mask_in = cp.array([0, 1, 0, 1], dtype=np.int8)
        >>> data_out = cp.empty_like(data_in)
        >>> offsets_out = cp.empty_like(offsets_in)
        >>> d_num_selected = cp.zeros(2, dtype=np.int32)
        >>> select_segments(data_in, offsets_in, mask_in, data_out, offsets_out,
        ...                 d_num_selected, len(data_in), len(offsets_in) - 1)
        >>> # Result: data_out contains [20, 50, 90, 20, 60, ...]
        >>> #         offsets_out contains [0, 3, 5, ...]
        >>> #         d_num_selected[0] = 5 (number of data elements)
        >>> #         d_num_selected[1] = 2 (number of segments kept)

    Args:
        data_in: Device array or iterator containing all segment elements concatenated.
        offsets_in: Device array or iterator of segment offsets (length = num_segments + 1).
                    Each segment i contains elements from offsets[i] to offsets[i+1].
        mask_in: Device array or iterator (int8) indicating which segments to keep (length = num_segments).
                 Non-zero values indicate segments to keep.
        data_out: Device array or iterator to store selected data elements.
                  Should be pre-allocated with size >= num_elements.
        offsets_out: Device array or iterator to store new segment offsets.
                     Should be pre-allocated with size >= num_segments + 1.
        d_num_selected_out: Device array to store counts (length >= 2):
                           - d_num_selected_out[0]: number of selected data elements
                           - d_num_selected_out[1]: number of segments kept
        num_elements: Total number of elements in data_in.
        num_segments: Total number of segments (= len(offsets_in) - 1).
        stream: CUDA stream for the operation (optional).
    """
    # Early exit for empty data
    if num_elements == 0:
        d_num_selected_out[0] = 0
        d_num_selected_out[1] = 0
        return

    # Step 1: Create segment_indices array indicating which segment each element belongs to
    segment_indices = offsets_to_segment_ids(offsets_in, stream)

    # Step 2: Expand mask from per-segment to per-element using PermutationIterator
    # Each element gets the mask value of its corresponding segment
    expanded_mask_it = PermutationIterator(mask_in, segment_indices)

    # Step 3: Filter the data array using the expanded mask
    # We only need to zip data with the expanded mask, no need for indices
    data_mask_in = ZipIterator(data_in, expanded_mask_it)
    d_num_data_selected = cp.zeros(1, dtype=np.int32)

    # Define predicate that checks if mask value is non-zero
    def mask_predicate(pair):
        return pair[1] != 0

    # Apply select to get filtered data
    select(
        data_mask_in,
        data_out,
        d_num_data_selected,
        mask_predicate,
        num_elements,
        stream,
    )

    num_selected = int(d_num_data_selected[0])

    # Step 4: Compute new segment offsets directly from mask and original offsets

    mask_array = cp.asarray(mask_in)
    kept_mask = mask_array != 0

    # Get segment sizes for ALL segments
    segment_sizes = offsets_in[1:] - offsets_in[:-1]

    # Filter to only kept segment sizes
    kept_segment_sizes = segment_sizes[kept_mask]
    num_kept_segments = len(kept_segment_sizes)

    # Exclusive scan to convert sizes to offsets
    if num_kept_segments > 0:
        h_init_scan = np.array([0], dtype=np.int32)
        exclusive_scan(
            kept_segment_sizes,
            offsets_out[: num_kept_segments + 1],
            OpKind.PLUS,
            h_init_scan,
            num_kept_segments,
            stream,
        )
        # The scan sets offsets_out[0:num_kept_segments] and we verify
        # offsets_out[num_kept_segments] should equal num_selected

    # Store the counts
    d_num_selected_out[0] = num_selected  # number of data elements
    d_num_selected_out[1] = num_kept_segments  # number of segments kept


@nvtx.annotate("segmented_select")
def segmented_select(
    d_in_data,
    d_in_segments,
    d_out_data,
    d_out_segments,
    cond,
    num_items: int,
    stream=None,
) -> int:
    """
    Select data within segments independently based on a condition.

    Given segmented input data and a selection condition, this function
    applies the selection to each segment independently and produces compacted
    output with updated segment offsets.

    Args:
        d_in_data: Device array containing the input data items.
        d_in_segments: Device array of segment offsets. For N segments,
            this array has N+1 elements where segments[i:i+1] defines
            the range [start, end) for segment i.
        d_out_data: Device array to store selected data (pre-allocated,
            should be at least as large as d_in_data).
        d_out_segments: Device array to store output segment offsets
            (pre-allocated, same size as d_in_segments).
        cond: Callable that takes a data item and returns a boolean-like
            value (typically uint8) indicating whether to keep the item.
        num_items: Total number of items in d_in_data.
        stream: CUDA stream for the operation (optional).

    Returns:
        int: Total number of items after selection (equal to d_out_segments[-1]).

    Example:
        >>> # Input: [[45], [25, 35], [15]] with condition x > 30
        >>> # Output: [[45], [35], []] -> offsets [0, 1, 2, 2]
        >>> def greater_than_30(x):
        ...     return x > 30
        >>> d_in_data = cp.array([45, 25, 35, 15], dtype=cp.int32)
        >>> d_in_segments = cp.array([0, 1, 3, 4], dtype=cp.int32)
        >>> d_out_data = cp.empty_like(d_in_data)
        >>> d_out_segments = cp.empty_like(d_in_segments)
        >>> total = segmented_select(
        ...     d_in_data, d_in_segments, d_out_data, d_out_segments,
        ...     greater_than_30, len(d_in_data)
        ... )
        >>> print(total)  # 2
        >>> print(d_out_segments.get())  # [0, 1, 2, 2]
    """
    import numba.cuda

    num_segments = len(d_in_segments) - 1

    cond = numba.cuda.jit(cond)
    # Apply select to get the data and indices where condition is true

    def select_predicate(pair):
        return cond(pair[0])

    data_and_indices_in = ZipIterator(d_in_data, CountingIterator(np.int32(0)))
    d_indices_out = cp.empty(num_items, dtype=np.int32)
    data_and_indices_out = ZipIterator(d_out_data, d_indices_out)
    d_num_selected = cp.zeros(1, dtype=cp.uint64)
    select(
        data_and_indices_in,
        data_and_indices_out,
        d_num_selected,
        select_predicate,
        num_items,
        stream,
    )
    total_selected = int(d_num_selected[0])
    d_indices_out = d_indices_out[:total_selected]
    d_selected_indices = d_indices_out[:total_selected]

    # Step 3: Use searchsorted to count selected items per segment
    # Use side='left' to count elements strictly less than each offset boundary
    positions = cp.searchsorted(d_selected_indices, d_in_segments, side="left")
    d_counts = (positions[1:] - positions[:-1]).astype(cp.uint64)

    # Step 4: Use exclusive scan to compute output segment start offsets
    exclusive_scan(
        d_counts,
        d_out_segments[:-1],
        OpKind.PLUS,
        np.array(0, dtype=np.uint64),
        num_segments,
        stream,
    )

    # Step 5: Set the final offset to the total count
    d_out_segments[-1] = total_selected
    return total_selected


_segmented_select_stateful_offsets = None
_segmented_select_stateful_removed = None
_segmented_select_stateful_num_segments = None
_segmented_select_stateful_cond = None


@numba.cuda.jit(device=True)
def _segmented_select_stateful_find_segment_id(offsets, num_segments, value):
    left = 0
    right = num_segments + 1
    while left < right:
        mid = (left + right) // 2
        if offsets[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left - 1


def _segmented_select_stateful_predicate(pair):
    value = pair[0]
    index = pair[1]
    keep = _segmented_select_stateful_cond(value)
    if not keep:
        num_segments = _segmented_select_stateful_num_segments[0]
        seg_id = _segmented_select_stateful_find_segment_id(
            _segmented_select_stateful_offsets, num_segments, index + 1
        )
        numba.cuda.atomic.add(_segmented_select_stateful_removed, seg_id, 1)
    return keep


@nvtx.annotate("segmented_select_stateful")
def segmented_select_stateful(
    d_in_data,
    d_in_segments,
    d_out_data,
    d_out_segments,
    cond,
    num_items: int,
    stream=None,
) -> int:
    """
    Select data within segments independently using a stateful predicate.

    This version mirrors the stateful approach in the C++ benchmark by
    tracking per-segment removals inside the predicate, then rebuilding
    output offsets from the original segment sizes and removed counts.
    """
    global _segmented_select_stateful_offsets
    global _segmented_select_stateful_removed
    global _segmented_select_stateful_num_segments
    global _segmented_select_stateful_cond

    num_segments = len(d_in_segments) - 1

    _segmented_select_stateful_cond = numba.cuda.jit(cond)

    if _segmented_select_stateful_num_segments is None:
        _segmented_select_stateful_num_segments = cp.zeros(1, dtype=np.int32)

    if (
        _segmented_select_stateful_offsets is None
        or _segmented_select_stateful_offsets.size < num_segments + 1
        or _segmented_select_stateful_offsets.dtype != d_in_segments.dtype
    ):
        _segmented_select_stateful_offsets = cp.empty(
            num_segments + 1, dtype=d_in_segments.dtype
        )
    if (
        _segmented_select_stateful_removed is None
        or _segmented_select_stateful_removed.size < num_segments
    ):
        _segmented_select_stateful_removed = cp.empty(num_segments, dtype=np.int64)

    _segmented_select_stateful_num_segments[0] = num_segments
    _segmented_select_stateful_offsets[: num_segments + 1] = d_in_segments
    _segmented_select_stateful_removed[:num_segments].fill(0)

    data_and_indices_in = ZipIterator(d_in_data, CountingIterator(np.int32(0)))
    d_num_selected = cp.zeros(1, dtype=cp.uint64)
    select(
        data_and_indices_in,
        d_out_data,
        d_num_selected,
        _segmented_select_stateful_predicate,
        num_items,
        stream,
    )

    total_selected = int(d_num_selected[0])

    segment_sizes = d_in_segments[1:] - d_in_segments[:-1]
    new_segment_sizes = (
        segment_sizes - _segmented_select_stateful_removed[:num_segments]
    )

    exclusive_scan(
        new_segment_sizes,
        d_out_segments[:-1],
        OpKind.PLUS,
        np.array(0, dtype=np.int64),
        num_segments,
        stream,
    )

    d_out_segments[-1] = total_selected
    return total_selected


@nvtx.annotate("transform_segments")
def transform_segments(data_in, data_out, segment_size, op, num_segments):
    """
    Apply the given n-ary operation to each segment, assuming the size of
    each segment is `n`.

    For example, if the segmented array is:
    [[1, 2], [4, 5], [7, 8]]
    and the operation is to add the two elements of each segment, the output will be:
    [[3], [9], [15]]

    Args:
        data_in: Device array containing the input data items.
        data_out: Device array to store the output data items.
        segment_size: Size of each segment.
        op: Operation to apply to each segment.
        num_segments: Number of segments.

    Args:
        data_in: Device array containing the input data items.
        data_out: Device array to store the output data items.
        segment_size: Size of each segment.
    """

    def column_iterator_factory(i):
        # closure avoids caching issues
        def column_iterator(j: np.int32) -> np.int32:
            return j * segment_size + i

        return column_iterator

    columns = ZipIterator(
        *[
            PermutationIterator(
                data_in,
                TransformIterator(
                    CountingIterator(np.int32(0)), column_iterator_factory(i)
                ),
            )
            for i in range(segment_size)
        ]
    )

    unary_transform(columns, data_out, op, num_segments)
