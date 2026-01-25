#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmark for differ().where() pattern using cuda.compute three_way_partition
with lazy iterators (matching parrot-style computation)
"""

import sys

import cuda.bench as bench
import cuda.compute.algorithms._three_way_partition as twp
import cuda.compute.iterators as iterators
import cupy as cp
import numpy as np

# Predicate for first partition: select if differ result is 1


def select_first_part_op(t):
    return t[1] == np.int32(1)


# Predicate for second partition: dummy (never selects anything)


def select_second_part_op(t):
    return False


class CCCLStream:
    "Class to work around https://github.com/NVIDIA/cccl/issues/5144"

    def __init__(self, ptr):
        self._ptr = ptr

    def __cuda_stream__(self):
        return (0, self._ptr)


def as_cccl_Stream(cs: bench.CudaStream) -> CCCLStream:
    return CCCLStream(cs.addressof())


def bench_differ_where(state: bench.State):
    """
    Benchmark the differ().where() pattern:
    - Takes a lazy iterator that produces random values
    - Computes adjacent differences (differ)
    - Partitions based on the differ result (where)
    """
    n = state.get_int64("Elements")

    # Output array - only first_part gets real storage (selected indices)
    # Second part and unselected use discard iterators (dummy, never used)
    first_part_indices = cp.empty(n, dtype=np.int32)

    # Wrap outputs as iterators (ZipIterator with DiscardIterator for the value)
    out_first = iterators.ZipIterator(first_part_indices, iterators.DiscardIterator())
    # Discard iterators for second part and unselected (not used in select_if pattern)
    out_second = iterators.ZipIterator(
        iterators.DiscardIterator(), iterators.DiscardIterator()
    )
    out_unselected = iterators.ZipIterator(
        iterators.DiscardIterator(), iterators.DiscardIterator()
    )

    # ============================================================================
    # OPTION A: Array-based approach (uncomment to use)
    # ============================================================================
    # with cp_stream:
    #     np.random.seed(42)
    #     sushi_data = cp.array(np.random.randint(0, 2, n, dtype=np.int32))

    # perm_a = iterators.PermutationIterator(
    #     sushi_data, iterators.CountingIterator(np.int32(0)))
    # perm_b = iterators.PermutationIterator(
    #     sushi_data, iterators.CountingIterator(np.int32(1)))
    # zip_adj = iterators.ZipIterator(perm_a, perm_b)

    # def neq_op(t):
    #     return np.int32(1) if t[0] != t[1] else np.int32(0)

    # differ_iter = iterators.TransformIterator(zip_adj, neq_op)
    # in_iter = iterators.ZipIterator(
    #     iterators.CountingIterator(np.int32(0)), differ_iter)
    # ============================================================================
    # END OPTION A
    # ============================================================================

    # ============================================================================
    # OPTION B: Parrot-style lazy iterator approach (currently active)
    # Matches: constant(2, n).rand().differ().where()
    # ============================================================================
    _entropy = np.uint32(0)

    # rand_op: generates pseudo-random value from (index, constant_val) tuple
    # Returns value in range [0, constant_val)
    def rand_op(t) -> np.int32:
        idx = np.int32(t[0])
        val = np.int32(t[1])

        # Hash (matches parrot's rand_op)
        h = np.uint32(idx ^ _entropy)
        h = np.uint32(((h >> 16) ^ h) * 0x45D9F3B)
        h = np.uint32(((h >> 16) ^ h) * 0x45D9F3B)
        h = np.uint32((h >> 16) ^ h)

        # Convert to float in [0, 1), multiply by val
        rand_val = np.float32(h & 0x7FFFFF) / np.float32(0x800000)
        return np.int32(rand_val * np.float32(val))

    # Step 1: Create the base iterator for rand(): ZipIterator(CountingIterator, ConstantIterator(2))
    base_iter = iterators.ZipIterator(
        iterators.CountingIterator(np.int32(0)), iterators.ConstantIterator(np.int32(2))
    )

    # Step 2: Apply rand_op to generate random values: TransformIterator(base_iter, rand_op)
    rand_iter = iterators.TransformIterator(base_iter, rand_op)

    # Step 3: Create PermutationIterators for differ() - accessing values[i] and values[i+1]
    perm_a = iterators.PermutationIterator(
        rand_iter, iterators.CountingIterator(np.int32(0))
    )
    perm_b = iterators.PermutationIterator(
        rand_iter, iterators.CountingIterator(np.int32(1))
    )

    # Step 4: Zip the two permutation iterators for adjacent comparison
    zip_adj = iterators.ZipIterator(perm_a, perm_b)

    # neq_op: returns 1 if tuple elements differ, 0 otherwise
    def neq_op(t):
        return np.int32(1) if t[0] != t[1] else np.int32(0)

    # Step 5: TransformIterator applies neq_op to each tuple from zip_adj
    differ_iter = iterators.TransformIterator(zip_adj, neq_op)

    # Step 6: ZipIterator combines index (starting at 0) with the differ result
    in_iter = iterators.ZipIterator(
        iterators.CountingIterator(np.int32(0)), differ_iter
    )
    # ============================================================================
    # END OPTION B
    # ============================================================================

    # Number of selected items output (2-element array: [num_first, num_second])
    d_num_selected = cp.empty(2, dtype=np.int32)

    # Build the three_way_partition algorithm
    alg = twp.make_three_way_partition(
        d_in=in_iter,
        d_first_part_out=out_first,
        d_second_part_out=out_second,
        d_unselected_out=out_unselected,
        d_num_selected_out=d_num_selected,
        select_first_part_op=select_first_part_op,
        select_second_part_op=select_second_part_op,
    )

    # Pre-allocate temporary storage (call with None first to get size)
    temp_storage_bytes = alg(
        None,
        in_iter,
        out_first,
        out_second,
        out_unselected,
        d_num_selected,
        n - 1,  # n-1 because differ() produces n-1 elements
        None,
    )
    temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    # Define the launcher function
    def launcher(launch: bench.Launch):
        s = as_cccl_Stream(launch.get_stream())
        alg(
            temp_storage,
            in_iter,
            out_first,
            out_second,
            out_unselected,
            d_num_selected,
            n - 1,  # n-1 because differ() produces n-1 elements
            s,
        )

    # Run the benchmark
    state.exec(launcher, sync=False)


if __name__ == "__main__":
    b = bench.register(bench_differ_where)
    b.add_int64_axis("Elements", [100000000])
    bench.run_all_benchmarks(sys.argv)
