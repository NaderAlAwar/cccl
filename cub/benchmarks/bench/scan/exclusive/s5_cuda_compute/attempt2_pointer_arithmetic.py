#!/usr/bin/env python3
"""
S5 2D Scan - Attempt 2: Pointer Arithmetic

This approach attempts to work around numba's restriction by using pointer arithmetic
instead of returning array slices. The idea is to return raw pointers to row starts,
similar to how the C++ version uses IndexToPointerFunctor.

**APPROACH:**
Instead of:
    def get_row(idx):
        return array[idx]  # Returns array - NOT ALLOWED

Try:
    def get_row_ptr(idx):
        # Return pointer to start of row
        # Requires working with ctypes pointers or numba's raw memory access

**CHALLENGES:**
1. Need to expose raw device pointers in a way numba can compile
2. Need value type that can load from pointer in dereference
3. cuda.compute infrastructure may not support pointer value types
"""

from typing import Tuple

import cuda.compute
import numba
import numba.cuda
import numpy as np
from cuda.compute import (
    CountingIterator,
    TransformIterator,
    TransformOutputIterator,
    ZipIterator,
)


def s5_scan_2d_pointer_arithmetic(
    A_in: np.ndarray, Bu_in: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt S5 scan using pointer arithmetic to avoid returning arrays.

    Strategy:
    1. TransformIterator returns POINTERS to row starts (not row values)
    2. These are uint64 addresses: base_ptr + idx * row_stride_bytes
    3. Operator dereferences pointers to load row data
    4. Performs S5 operation and writes back

    This mimics C++ IndexToPointerFunctor which returns T* pointers.

    Args:
        A_in: Input A values, shape (timesteps, state_dim)
        Bu_in: Input Bu values, shape (timesteps, state_dim)

    Returns:
        Tuple of (A_out, Bu_out) with shape (timesteps, state_dim) each
    """
    timesteps, state_dim = A_in.shape

    # Use numba device arrays
    d_A_in = numba.cuda.to_device(A_in)
    d_Bu_in = numba.cuda.to_device(Bu_in)
    d_A_out = numba.cuda.device_array((timesteps, state_dim), dtype=A_in.dtype)
    d_Bu_out = numba.cuda.device_array((timesteps, state_dim), dtype=Bu_in.dtype)

    # ========================================================================
    # Pointer Arithmetic Approach
    # ========================================================================

    # Get base pointers as integers (device memory addresses)
    # In numba, we can use .device_ctypes_pointer.value to get the raw address
    A_in_ptr = d_A_in.device_ctypes_pointer.value
    Bu_in_ptr = d_Bu_in.device_ctypes_pointer.value
    A_out_ptr = d_A_out.device_ctypes_pointer.value
    Bu_out_ptr = d_Bu_out.device_ctypes_pointer.value

    # Calculate stride in bytes
    dtype_size = A_in.dtype.itemsize
    row_stride_bytes = state_dim * dtype_size

    # Step 1: CountingIterator for row indices
    row_index_iter = CountingIterator(np.int32(0))

    # Step 2: TransformIterator that returns POINTERS (uint64), not arrays
    # This should compile since we're returning scalar uint64, not arrays
    def get_A_row_ptr(idx):
        """
        Returns pointer to start of row idx in A matrix.
        Returns uint64 address, not an array.
        """
        # Compute: base_ptr + idx * row_stride_bytes
        return np.uint64(A_in_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    def get_Bu_row_ptr(idx):
        """Returns pointer to start of row idx in Bu matrix."""
        return np.uint64(Bu_in_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    # These should compile - we're returning uint64 scalars
    A_ptr_iter = TransformIterator(row_index_iter, get_A_row_ptr)
    Bu_ptr_iter = TransformIterator(row_index_iter, get_Bu_row_ptr)

    # Step 3: Zip the pointer iterators
    # Each element is now a pair of pointers (uint64, uint64)
    _input_ptr_iter = ZipIterator(A_ptr_iter, Bu_ptr_iter)  # noqa: F841 (demo code)

    # ========================================================================
    # Output Iterator
    # ========================================================================

    row_index_iter_out = CountingIterator(np.int32(0))

    def get_A_out_row_ptr(idx):
        return np.uint64(A_out_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    def get_Bu_out_row_ptr(idx):
        return np.uint64(Bu_out_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    # Use TransformOutputIterator for output (not TransformIterator)
    A_out_ptr_iter = TransformOutputIterator(row_index_iter_out, get_A_out_row_ptr)
    Bu_out_ptr_iter = TransformOutputIterator(row_index_iter_out, get_Bu_out_row_ptr)

    output_ptr_iter = ZipIterator(A_out_ptr_iter, Bu_out_ptr_iter)

    # ========================================================================
    # S5 Operator with Pointer Dereference
    # ========================================================================

    # The challenge: The operator receives pairs of uint64 pointers
    # We need to:
    # 1. Dereference these pointers to load row data
    # 2. Perform S5 operations
    # 3. Store results back through output pointers
    #
    # But cuda.compute scan expects operators to work on VALUE types,
    # not perform manual memory loads/stores. This is where the approach
    # may still fail.

    _ZipValue = output_ptr_iter.value_struct_type  # noqa: F841 (would be used if implemented)

    def s5_op_2d_ptr(x_ptrs, y_ptrs):
        """
        S5 operator that works with pointer pairs.

        Args:
            x_ptrs: ZipValue(A_ptr, Bu_ptr) - uint64 pointers
            y_ptrs: ZipValue(A_ptr, Bu_ptr) - uint64 pointers

        Challenge: We need to dereference these pointers to arrays,
        but that brings us back to the array return problem.
        """
        # Extract pointers (intentionally unused - demonstrates the problem)
        _x_A_ptr = x_ptrs[0]  # uint64  # noqa: F841
        _x_Bu_ptr = x_ptrs[1]  # uint64  # noqa: F841
        _y_A_ptr = y_ptrs[0]  # uint64  # noqa: F841
        _y_Bu_ptr = y_ptrs[1]  # uint64  # noqa: F841

        # How do we dereference these pointers to get arrays?
        # We'd need something like:
        #   x_A = pointer_to_array(x_A_ptr, state_dim, dtype)
        #
        # But this is complex in numba and may not be supported
        # by cuda.compute's operator compilation.

        # Placeholder - this won't work without proper pointer dereferencing
        raise NotImplementedError(
            "Pointer dereferencing in operator not yet implemented. "
            "Need infrastructure to convert uint64 pointers back to arrays "
            "within compiled device code."
        )

    # This won't work without proper operator implementation
    # cuda.compute.inclusive_scan(
    #     d_in=input_ptr_iter,
    #     d_out=output_ptr_iter,
    #     op=s5_op_2d_ptr,
    #     init_value=None,
    #     num_items=timesteps,
    # )

    A_out = d_A_out.copy_to_host()
    Bu_out = d_Bu_out.copy_to_host()

    return A_out, Bu_out


# ============================================================================
# Test/Demo Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Attempt 2: Pointer Arithmetic")
    print("=" * 80)
    print("\nThis approach has two phases:")
    print("  Phase 1: Create iterators that return pointers (uint64)")
    print("  Phase 2: Operator dereferences pointers to get arrays")
    print("\n")

    timesteps = 8
    state_dim = 4

    A_in = np.random.randn(timesteps, state_dim).astype(np.float32)
    Bu_in = np.random.randn(timesteps, state_dim).astype(np.float32)

    print(f"Input shape: ({timesteps}, {state_dim})")
    print("\n" + "-" * 80)
    print("Phase 1: Creating pointer-based iterators...")
    print("-" * 80)

    # Use numba device arrays
    d_A_in = numba.cuda.to_device(A_in)
    d_Bu_in = numba.cuda.to_device(Bu_in)
    d_A_out = numba.cuda.device_array((timesteps, state_dim), dtype=A_in.dtype)
    d_Bu_out = numba.cuda.device_array((timesteps, state_dim), dtype=Bu_in.dtype)

    # Get base pointers
    A_in_ptr = d_A_in.device_ctypes_pointer.value
    Bu_in_ptr = d_Bu_in.device_ctypes_pointer.value

    dtype_size = A_in.dtype.itemsize
    row_stride_bytes = state_dim * dtype_size

    # Create counting iterator
    row_index_iter = CountingIterator(np.int32(0))

    # Create transform functions that return pointers (uint64)
    def get_A_row_ptr(idx):
        return np.uint64(A_in_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    def get_Bu_row_ptr(idx):
        return np.uint64(Bu_in_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    # Try to create the transform iterators
    print("  Creating TransformIterator that returns uint64 pointers...")
    A_ptr_iter = TransformIterator(row_index_iter, get_A_row_ptr)
    Bu_ptr_iter = TransformIterator(row_index_iter, get_Bu_row_ptr)

    print("  ✓ Phase 1 SUCCESS!")
    print("    TransformIterator successfully created with uint64 return type")
    print("    This works because we're returning scalars (pointers), not arrays")

    print("\n" + "-" * 80)
    print("Phase 2: Attempting to create operator that dereferences pointers...")
    print("-" * 80)
    print("  Creating an operator that loads arrays from uint64 pointers...\n")

    # Create output iterators
    # Extract pointers on HOST first (before function definitions)
    A_out_ptr = d_A_out.device_ctypes_pointer.value
    Bu_out_ptr = d_Bu_out.device_ctypes_pointer.value

    row_index_iter_out = CountingIterator(np.int32(0))

    # Now the pointer values are captured as integers, not accessed during compilation
    # TransformOutputIterator requires type annotations
    def get_A_out_row_ptr(idx: np.uint64) -> np.uint64:
        return np.uint64(A_out_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    def get_Bu_out_row_ptr(idx: np.uint64) -> np.uint64:
        return np.uint64(Bu_out_ptr) + np.uint64(idx) * np.uint64(row_stride_bytes)

    # Use TransformOutputIterator for output (not TransformIterator)
    A_out_ptr_iter = TransformOutputIterator(row_index_iter_out, get_A_out_row_ptr)
    Bu_out_ptr_iter = TransformOutputIterator(row_index_iter_out, get_Bu_out_row_ptr)

    # Zip the pointer iterators (A_ptr_iter and Bu_ptr_iter already created in Phase 1)
    input_ptr_iter = ZipIterator(A_ptr_iter, Bu_ptr_iter)
    output_ptr_iter = ZipIterator(A_out_ptr_iter, Bu_out_ptr_iter)

    # Define operator that tries to dereference pointers
    # This will fail when TransformIterator or scan tries to compile it
    def s5_op_ptr(x_ptrs, y_ptrs):
        """
        Operator that attempts to dereference uint64 pointers to arrays.
        This will fail!
        """
        # Extract pointers (these are uint64 values)
        _x_A_ptr = x_ptrs[0]  # noqa: F841 (demo code)

        # Try to load array from pointer - THIS IS WHERE IT WILL FAIL
        # We need to convert uint64 pointer to array view somehow
        # Let's try to access it as if it were already an array (will fail)

        # Attempt to create array from pointer - numba won't allow this
        _x_A = d_A_in[0]  # This returns an array, which numba won't allow  # noqa: F841

        # Return a tuple (will fail before getting here anyway)
        return (np.uint64(0), np.uint64(0))

    print("  Attempting scan with pointer dereferencing operator...")
    # This will fail when it tries to compile the operator
    cuda.compute.inclusive_scan(
        d_in=input_ptr_iter,
        d_out=output_ptr_iter,
        op=s5_op_ptr,
        init_value=None,
        num_items=timesteps,
    )

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
Pointer arithmetic successfully moves the problem from TransformIterator to
the operator, but doesn't solve the fundamental issue:

  Numba CUDA won't let you work with arrays in device functions
  unless they're passed as arguments.

To truly work around this, we'd need:
- Custom iterator infrastructure that handles array indexing internally
- Or different algorithm that doesn't require row-level iteration
""")


# ============================================================================
# Alternative: Manual Pointer Dereferencing Helper
# ============================================================================


def _create_pointer_to_array_helper(state_dim: int, dtype):
    """
    Attempt to create a helper that converts pointers to arrays in device code.

    This is complex because:
    1. Need to work with raw pointers in numba device code
    2. Need to create array views from pointers
    3. Must be compilable by numba CUDA
    """

    @numba.cuda.jit(device=True)
    def load_row_from_ptr(ptr_uint64):
        """
        Load a row from a device pointer.

        Args:
            ptr_uint64: uint64 device pointer to row start

        Returns:
            Array of length state_dim - BUT THIS IS THE PROBLEM!
        """
        # We'd need something like:
        # array = numba.cuda.from_device_pointer(ptr_uint64, shape=(state_dim,), dtype=dtype)
        #
        # But returning arrays from device functions is not allowed by numba.
        # This brings us back to the original problem.
        pass

    return load_row_from_ptr


# ============================================================================
# Conclusion
# ============================================================================
#
# The pointer arithmetic approach successfully avoids the array return problem
# in the TransformIterator (by returning uint64 pointers instead).
#
# However, we hit the same fundamental limitation in the operator:
# - The operator needs to dereference pointers to get row data
# - This requires returning/working with arrays in device code
# - Numba doesn't allow this
#
# To truly solve this, we need one of:
# 1. Custom iterator that handles 2D indexing internally (strided iterator approach)
# 2. cuda.compute infrastructure changes to support pointer value types with
#    automatic dereferencing
# 3. Different algorithm that doesn't require row-level operations
