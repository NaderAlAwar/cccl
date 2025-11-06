#!/usr/bin/env python3
"""
S5 2D Scan - Attempt 1: TransformIterator with Row Extraction

This approach attempts to use TransformIterator to extract rows from 2D arrays,
mapping each timestep index to a full row vector.

**STATUS: DOES NOT WORK**

**LIMITATION:**
The fundamental issue is that numba CUDA restricts device functions from returning
arrays unless they were passed in as arguments. When TransformIterator tries to
compile `get_A_row(idx) -> d_A_in[idx]`, numba raises:
    "Only accept returning of array passed into the function as argument"

This means we cannot use TransformIterator to extract rows from 2D arrays, which
is the core requirement for treating rows as "elements" in a scan.

**Why C++ works but Python doesn't:**
1. C++ transform_iterator can return pointers (not array values)
2. C++ VectorPair loads data in its constructor (different memory model)
3. Thrust's iterator composition is more flexible with value types
"""

from typing import Tuple

import cuda.compute
import numba.cuda
import numpy as np
from cuda.compute import (
    CountingIterator,
    TransformIterator,
    ZipIterator,
)


def s5_scan_2d_transform_iterator(
    A_in: np.ndarray, Bu_in: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to perform S5 scan on 2D arrays using TransformIterator.

    Strategy (mirroring C++ s5_operator.cu):
    1. CountingIterator generates row indices: 0, 1, 2, ..., timesteps-1
    2. TransformIterator maps index -> row vector  ❌ FAILS HERE
    3. ZipIterator pairs (A_row, Bu_row)
    4. S5 operator performs elementwise operations on row pairs
    5. Single scan over timesteps (not timesteps * state_dim)

    Args:
        A_in: Input A values, shape (timesteps, state_dim)
        Bu_in: Input Bu values, shape (timesteps, state_dim)

    Returns:
        Tuple of (A_out, Bu_out) with shape (timesteps, state_dim) each

    Raises:
        NumbaTypeError: "Only accept returning of array passed into the function as argument"
    """
    timesteps, state_dim = A_in.shape

    # Use numba device arrays (TransformIterator needs numba-compatible types)
    d_A_in = numba.cuda.to_device(A_in)
    d_Bu_in = numba.cuda.to_device(Bu_in)
    d_A_out = numba.cuda.device_array((timesteps, state_dim), dtype=A_in.dtype)
    d_Bu_out = numba.cuda.device_array((timesteps, state_dim), dtype=Bu_in.dtype)

    # ========================================================================
    # Input Iterator Composition
    # ========================================================================

    # Step 1: CountingIterator generates row indices
    row_index_iter = CountingIterator(np.int32(0))

    # Step 2: TransformIterator maps index -> row
    # This is where it FAILS - numba won't allow returning arrays
    def get_A_row(idx):
        """Extract row idx from A matrix."""
        return d_A_in[idx]  # ❌ Error: "Only accept returning of array..."

    def get_Bu_row(idx):
        """Extract row idx from Bu matrix."""
        return d_Bu_in[idx]  # ❌ Error: "Only accept returning of array..."

    # This will fail when TransformIterator tries to compile these functions
    A_row_iter = TransformIterator(row_index_iter, get_A_row)
    Bu_row_iter = TransformIterator(row_index_iter, get_Bu_row)

    # Step 3: ZipIterator pairs A and Bu rows
    input_iter = ZipIterator(A_row_iter, Bu_row_iter)

    # ========================================================================
    # Output Iterator Composition
    # ========================================================================

    row_index_iter_out = CountingIterator(np.int32(0))

    def set_A_row(idx):
        return d_A_out[idx]

    def set_Bu_row(idx):
        return d_Bu_out[idx]

    A_out_row_iter = TransformIterator(row_index_iter_out, set_A_row)
    Bu_out_row_iter = TransformIterator(row_index_iter_out, set_Bu_row)

    output_iter = ZipIterator(A_out_row_iter, Bu_out_row_iter)

    # ========================================================================
    # S5 Operator for Vector Pairs
    # ========================================================================

    ZipValue = output_iter.value_struct_type

    def s5_op_2d(x, y):
        """S5 operator for vector pairs with elementwise operations."""
        x_A = x[0]
        x_Bu = x[1]
        y_A = y[0]
        y_Bu = y[1]

        result_A = y_A * x_A
        result_Bu = y_A * x_Bu + y_Bu

        return ZipValue(result_A, result_Bu)

    # ========================================================================
    # Single Scan Call
    # ========================================================================

    cuda.compute.inclusive_scan(
        d_in=input_iter,
        d_out=output_iter,
        op=s5_op_2d,
        init_value=None,
        num_items=timesteps,
    )

    A_out = d_A_out.copy_to_host()
    Bu_out = d_Bu_out.copy_to_host()

    return A_out, Bu_out


# ============================================================================
# Test/Demo Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Attempt 1: TransformIterator with Row Extraction")
    print("=" * 80)
    print("\nThis will fail with:")
    print(
        '  NumbaTypeError: "Only accept returning of array passed into the function as argument"'
    )
    print("\nGenerating test data...")

    timesteps = 8
    state_dim = 4

    A_in = np.random.randn(timesteps, state_dim).astype(np.float32)
    Bu_in = np.random.randn(timesteps, state_dim).astype(np.float32)

    print(f"  Input shape: ({timesteps}, {state_dim})")
    print("  Attempting to create iterators...\n")

    # Call the function - will raise NumbaTypeError
    A_out, Bu_out = s5_scan_2d_transform_iterator(A_in, Bu_in)
